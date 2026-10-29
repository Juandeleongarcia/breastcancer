"""Resultados CV de solo lectura. Nunca llama al resumidor que escribe campañas."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app_services.workspace import local_path, ROOT
from src.cv_data import audit_samples, fold_frames, artifact_hash
from src.cross_validation import aggregate_oof
from src.metrics import binary_metrics
from src.run_layout import technical_file


@dataclass
class FoldResult:
    fold: int
    status: str = "Pendiente"
    complete: bool = False
    issue: str = ""
    run: Path | None = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class CampaignReport:
    root: Path
    architecture: str
    template: dict
    folds: list
    issue: str = ""
    oof: pd.DataFrame = field(default_factory=pd.DataFrame)
    oof_metrics: dict = field(default_factory=dict)

    @property
    def completed(self):
        return [fold for fold in self.folds if fold.complete]

    @property
    def complete(self):
        return len(self.completed) == 5 and not self.oof.empty

    @property
    def metrics(self):
        done = self.completed
        if not done: return dict(complete_folds=0,patients=0)
        scores = [fold.metrics['auc'] for fold in done]
        return dict(complete_folds=len(done),patients=sum(len(f.predictions) for f in done),
                    auc_mean=float(np.mean(scores)),auc_std=float(np.std(scores,ddof=1)) if len(done)>1 else None,
                    accuracy=float(np.mean([f.metrics['accuracy'] for f in done])),
                    balanced_accuracy=float(np.mean([f.metrics['balanced_accuracy'] for f in done])),
                    oof_auc=self.oof_metrics.get('auc'))


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:data = json.load(stream)
    if not isinstance(data,dict):raise ValueError('Se esperaba un objeto JSON: '+Path(path).name)
    return data


def _campaign_architecture(config, fallback):
    """Leer métricas no requiere disponer de una factory o de pesos de esa CNN."""
    name = config.get('ARCHITECTURE') or fallback
    if not isinstance(name,str) or not name.strip():
        raise ValueError('Falta identificación interna en la campaña')
    return name


def read_history(filename):
    history = pd.read_csv(filename)
    if 'epoch' not in history or history.empty:
        raise ValueError("El historial no contiene épocas")
    epochs = pd.to_numeric(history.epoch,errors='raise')
    if not np.isfinite(epochs).all() or not np.equal(epochs,np.floor(epochs)).all():
        raise ValueError("Épocas no enteras o no finitas")
    if list(epochs) != list(range(1,len(history)+1)):
        raise ValueError("El historial debe contener épocas consecutivas, sin duplicaciones")
    for name in history:
        if name.endswith(('_loss','_auc','_accuracy')) or name in ['learning_rate','next_learning_rate']:
            history[name] = pd.to_numeric(history[name],errors='raise')
            values = history[name].dropna()
            if not np.isfinite(values).all(): raise ValueError('Métricas no finitas: '+name)
            if name.endswith(('_auc','_accuracy')) and not values.between(0,1).all():
                raise ValueError('Métrica fuera de [0,1]: '+name)
            if name.endswith('_loss') and values.lt(0).any(): raise ValueError('Loss negativa')
    return history


def aggregate_histories(histories, mode='all', train_column='train_eval_loss'):
    """Estadísticas de filas existentes; sin relleno, suavizado ni interpolación.

    Una época contribuye si ambas pérdidas están presentes. DE muestral (ddof=1);
    queda NaN cuando solo hay un fold, sin inventar una banda de incertidumbre.
    """
    if mode not in ['all','common']: raise ValueError('Modo de agregación desconocido')
    if mode == 'common' and set(histories) != set(range(5)): return pd.DataFrame()
    values = []
    for fold, history in histories.items():
        if not {train_column,'val_loss','epoch'} <= set(history): continue
        rows = history[['epoch',train_column,'val_loss']].copy().dropna()
        rows['fold'] = fold
        values.append(rows.rename(columns={train_column:'train'}))
    if not values: return pd.DataFrame()
    data = pd.concat(values,ignore_index=True)
    if data.duplicated(['fold','epoch']).any(): raise ValueError('Épocas duplicadas por fold')
    if not np.isfinite(data[['train','val_loss']].to_numpy()).all(): raise ValueError('Pérdidas no finitas')
    output = data.groupby('epoch').agg(train_mean=('train','mean'),train_std=('train','std'),
             val_mean=('val_loss','mean'),val_std=('val_loss','std'),n_folds=('fold','nunique')).reset_index()
    return output[output.n_folds.eq(5)].reset_index(drop=True) if mode=='common' else output


def _run_for(root, fold, entry):
    folder = root/f'fold_{fold}'/'runs'
    saved = entry.get('last_run')
    if entry.get('status') == 'complete':
        if not saved: return None
        name = str(saved).replace('\\','/').rsplit('/',1)[-1]
        if not name.startswith('RUN_') or name in ['.','..']:
            raise ValueError('Nombre de RUN no válido')
        target = folder/name
        return target if target.is_dir() else None
    candidates = list(folder.glob('RUN_*'))
    return max(candidates,key=lambda p:p.stat().st_mtime_ns) if candidates else None


def _predictions(filename, validation):
    predictions = pd.read_csv(filename,dtype={'patient_id':str})
    required = {'patient_id','label','probability'}
    if not required <= set(predictions) or predictions[list(required)].isna().any().any():
        raise ValueError('Predicciones por paciente incompletas')
    if predictions.patient_id.duplicated().any() or not predictions.label.isin([0,1]).all():
        raise ValueError('Pacientes duplicados o etiquetas inválidas')
    probabilities = pd.to_numeric(predictions.probability,errors='raise')
    if not np.isfinite(probabilities).all() or not probabilities.between(0,1).all():
        raise ValueError('Probabilidades por paciente inválidas')
    expected = validation[['patient_id','pCR']].drop_duplicates()
    joined = predictions.merge(expected,on='patient_id',validate='one_to_one')
    if set(predictions.patient_id) != set(expected.patient_id) or not joined.label.eq(joined.pCR).all():
        raise ValueError('Predicciones no cubren exactamente los pacientes del fold')
    if 'n_slices' in predictions:
        counts = validation.groupby('patient_id').size()
        if not predictions.set_index('patient_id').n_slices.eq(counts).all():
            raise ValueError('El número de cortes por paciente no coincide con la validación')
    predictions['probability'] = probabilities
    return predictions


def read_campaign(directory, expected_architecture=None):
    root = local_path(directory)
    state = _json(root/'campaign.json')
    template = _json(root/'template.json')
    if not isinstance(state.get('folds'),dict) or set(state['folds']) != {str(i) for i in range(5)}:
        raise ValueError('El manifiesto debe definir exactamente los folds oficiales 0–4')
    kind = _campaign_architecture(template,state.get('model'))
    if expected_architecture and kind != expected_architecture:
        raise ValueError('La campaña corresponde a otra arquitectura')
    report = CampaignReport(root,kind,template,[])
    source = None
    try:
        candidate = local_path(state['protocol']['samples_csv'])
        if not candidate.is_file(): candidate = ROOT/'metadata/samples.csv'
        source,_ = audit_samples(candidate,expected_sha256=state['source_sha256'])
        if state.get('template_sha256') != artifact_hash(root/'template.json'):
            raise ValueError('El template ha cambiado respecto al manifiesto')
    except (OSError,ValueError,KeyError) as error:
        report.issue = 'No se ha podido verificar el protocolo oficial: '+str(error)
    for fold in range(5):
        item = FoldResult(fold)
        report.folds.append(item)
        entry = state.get('folds',{}).get(str(fold),{})
        item.status = {'prepared':'Pendiente','ready':'Preparado','running':'En ejecución',
                       'error':'Error','interrupted':'Interrumpido','complete':'Por verificar'}.get(entry.get('status'),'Pendiente')
        try:
            item.run = _run_for(root,fold,entry)
            if item.run is None:
                if entry.get('status') == 'complete': raise ValueError('Falta el RUN declarado completo')
                continue
            if (item.run/'history.csv').exists(): item.history = read_history(item.run/'history.csv')
            summary_file = technical_file(item.run,'summary.json')
            if not summary_file.is_file(): continue
            item.summary = _json(summary_file)
            config_file = technical_file(item.run,'config.json')
            item.config = _json(config_file)
            if _campaign_architecture(item.config,kind) != kind: raise ValueError('Arquitectura del RUN diferente al template')
            keys = ['CONV_CHANNELS','DROPOUT','LEARNING_RATE','WEIGHT_DECAY','BATCH_SIZE','SEED',
                    'EPOCHS','PATIENCE','OPTIMIZER','CONVS_PER_BLOCK','USE_WEIGHTED_LOSS','INPUT_MODE',
                    'USE_CLIPPING','USE_STANDARDIZATION','SCHEDULER','SCHEDULER_PATIENCE','EVALUATE_TRAIN',
                    'USE_AUGMENTATION','AUGMENTATION_MODE','IMAGE_SIZE','INPUT_CHANNELS','SPATIAL_GRID',
                    'KERNEL_SIZE','PADDING','POOL_SIZE','USE_BATCH_NORM','SCHEDULER_FACTOR','SCHEDULER_MIN_LR',
                    'CLASSIFICATION_THRESHOLD','PRIMARY_METRIC','ARCHITECTURE','BLOCK_CHANNELS',
                    'POOL_AFTER_BLOCK','CONV_BIAS']
            if any(item.config.get(key) != template.get(key) for key in keys):
                raise ValueError('La configuración del RUN difiere del template de campaña')
            if source is None or report.issue: raise ValueError(report.issue)
            expected_train, expected_validation = fold_frames(source,fold)
            folder = root/f'fold_{fold}'
            training = pd.read_csv(folder/'train.csv')
            validation = pd.read_csv(folder/'validation.csv')
            if not training.equals(expected_train) or not validation.equals(expected_validation):
                raise ValueError('Los CSV derivados difieren de los folds oficiales')
            for name,field_name in [('train.csv','train_sha256'),('validation.csv','validation_sha256')]:
                if entry.get(field_name) != artifact_hash(folder/name): raise ValueError('Partición alterada: '+name)
            for name,digest in entry.get('artifacts',{}).items():
                if Path(name).name != name or artifact_hash(folder/name) != digest:
                    raise ValueError('Configuración/estadísticas derivadas alteradas')
            item.predictions = _predictions(technical_file(item.run,'predictions_patient.csv'),validation)
            threshold = float(item.config['CLASSIFICATION_THRESHOLD'])
            if not 0 <= threshold <= 1: raise ValueError('Umbral inválido')
            item.metrics = binary_metrics(item.predictions.label,item.predictions.probability,threshold)
            summary = item.summary
            epochs = int(summary['epochs_completed'])
            best = int(summary['best_epoch'])
            if epochs != len(item.history) or best not in set(item.history.epoch):
                raise ValueError('Resumen e historial incoherentes')
            score = float(item.history.set_index('epoch').loc[best,'val_patient_auc'])
            if not np.isclose(score,item.history.val_patient_auc.max(),atol=5e-6,rtol=0):
                raise ValueError('La mejor época declarada no corresponde al máximo AUC paciente')
            if not np.isclose(score,item.metrics['auc'],atol=5e-6,rtol=0):
                raise ValueError('Las predicciones no corresponden al AUC de la mejor época')
            if not np.isclose(summary['validation']['patient']['auc'],score,atol=5e-6,rtol=0):
                raise ValueError('AUC del resumen incoherente')
            diagnostic = bool(state.get('diagnostic') or summary.get('diagnostic_partial') or item.config.get('SMOKE_TEST')
                              or item.config.get('MAX_TRAIN_BATCHES') or item.config.get('MAX_VAL_BATCHES')
                              or item.config.get('TRAIN_PATIENTS_PER_CLASS'))
            if 'diagnostic_partial' in item.history:
                diagnostic |= item.history.diagnostic_partial.astype(str).str.lower().eq('true').any()
            ended = epochs >= item.config['EPOCHS'] or summary.get('stopped_early') is True
            item.complete = bool(entry.get('status')=='complete' and not diagnostic and ended
                                 and summary.get('status')=='entrenamiento_completo' and summary.get('validation_complete') is True)
            if item.complete: item.status = 'Completo'
            elif diagnostic: item.status = 'Diagnóstico · excluido de CV'
            else: item.status += ' · provisional'
        except (OSError,ValueError,KeyError,TypeError,pd.errors.ParserError) as error:
            item.complete = False
            item.status = 'No verificable'
            item.issue = str(error)
            item.metrics = {}
            item.predictions = pd.DataFrame()
    if len(report.completed)==5:
        try:
            thresholds = {f.config['CLASSIFICATION_THRESHOLD'] for f in report.completed}
            if len(thresholds)!=1: raise ValueError('Umbrales diferentes entre folds')
            pooled = pd.concat([f.predictions.assign(cv_fold=f.fold) for f in report.completed],ignore_index=True)
            aggregate_oof(source,pooled)  # cobertura exacta, paciente único y fold excluido correcto
            report.oof_metrics = binary_metrics(pooled.label,pooled.probability,thresholds.pop())
            report.oof = pooled
        except (ValueError,KeyError) as error:
            report.issue = 'OOF no verificable: '+str(error)
    return report
