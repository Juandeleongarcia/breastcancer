"""Análisis exploratorio OOF; nunca carga ni evalúa predicciones del test externo."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app_services.workspace import ROOT
from src.metrics import binary_metrics
from src.configuration import canonical_csv_hash
from src.cv_data import artifact_hash

GRID = tuple(i / 100 for i in range(5, 96))
WARNING = ('Análisis exploratorio de desarrollo: los checkpoints se seleccionaron por AUC '
           'sobre estos mismos folds. No es CV anidada ni una estimación independiente del '
           'rendimiento tras seleccionar el umbral. OOF usa un modelo no entrenado con cada '
           'paciente; no reproduce la distribución del ensemble de cinco modelos. La '
           'recomendación para nuevas muestras es provisional y no está validada en test.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decision(result, threshold=.5):
    """Cambiar solo la decisión; conservar intactas las probabilidades de inferencia."""
    if not np.isfinite(threshold) or not .05 <= threshold <= .95:
        raise ValueError('El umbral debe estar entre 0,05 y 0,95')
    return {**result, 'threshold': float(threshold),
            'prediction': int(result['patient_probability'] >= threshold)}


def analyze(labels, probabilities):
    labels=np.asarray(labels); probabilities=np.asarray(probabilities, dtype=float)
    if labels.ndim!=1 or probabilities.shape!=labels.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError('Se requieren ambas clases de desarrollo')
    rows=[]
    for threshold in GRID:
        metrics=binary_metrics(labels, probabilities, threshold)
        rows.append(dict(threshold=threshold, **metrics,
                         youden=metrics['sensitivity']+metrics['specificity']-1))
    maximum=max(row['youden'] for row in rows)
    tied=[row for row in rows if abs(row['youden']-maximum) <= 1e-12]
    # Regla fijada antes del análisis: menor distancia a 0,50; después mayor umbral.
    best=min(tied, key=lambda row: (round(abs(row['threshold']-.5), 12), -row['threshold']))
    return dict(schema_version=1, scope='multimodal_patient_oof_development_only',
        methodology=dict(positive_class='1 = pCR', aggregation='one held-out prediction per patient',
            decision_rule='probability >= threshold', objective='sensitivity + specificity - 1',
            grid=dict(min=.05,max=.95,step=.01,points=len(GRID)),
            tie_break='within 1e-12: closest to 0.50, then higher threshold',
            zero_division=0, warning=WARNING, external_test_used=False,
            nested_cross_validation=False, ensemble_transfer_validated=False),
        n=len(labels), positives=int((labels==1).sum()), negatives=int((labels==0).sum()),
        recommended_threshold=best['threshold'], recommended=best,
        original=next(row for row in rows if row['threshold']==.5), thresholds=rows)


def child(root, relative):
    path=(root/str(relative).replace('\\','/')).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Referencia de campaña no válida')
    return path


def recover_oof(campaign, samples_csv):
    """Verificar manifiestos, folds oficiales y OOF guardadas, sin acceder a test/."""
    campaign=Path(campaign).resolve(); samples_csv=Path(samples_csv)
    manifest=campaign/'campaign.json'; state=json.loads(manifest.read_text(encoding='utf-8'))
    if state.get('schema_version')!=13 or state.get('SMOKE_TEST') or state.get('architecture')!='SPATIAL_SMALL_MULTIMODAL':
        raise ValueError('Se requiere una campaña F13 multimodal completa')
    protocol=state['protocol']
    if protocol['allowed_split']!='train' or protocol['reserved_fold']!=-1 or protocol['folds']!=list(range(5)):
        raise ValueError('Protocolo oficial no compatible')
    if canonical_csv_hash(samples_csv)!=protocol['samples_sha256']:
        raise ValueError('Las particiones no coinciden con la campaña')
    # Solo se toman etiquetas/identidades de desarrollo; no se abren imágenes ni test predictions.
    samples=pd.read_csv(samples_csv)
    dev=samples.loc[samples.split.eq('train'), ['patient_id','fold','pCR']].drop_duplicates()
    if len(dev)!=1097 or dev.patient_id.duplicated().any() or set(dev.fold)!=set(range(5)):
        raise ValueError('Se esperan los 1.097 pacientes oficiales, sin duplicados')
    sources=[];frames=[];columns=['patient_id','label','probability','cv_fold']
    for fold in range(5):
        member=state['folds'][str(fold)]
        if member['status']!='complete':raise ValueError('Los cinco folds deben estar completados')
        run=child(campaign, member['last_run']+'/internals/summary.json').parent.parent
        pred=run/'internals/predictions_patient.csv'; summary=run/'internals/summary.json'
        for path in [pred, summary]:
            relative=path.relative_to(campaign).as_posix()
            if artifact_hash(path)!=member['result_artifacts'].get(relative):
                raise ValueError('Resultados de un fold modificados o no verificables')
        info=json.loads(summary.read_text(encoding='utf-8'))
        if info.get('status')!='complete' or not info.get('validation_complete') or info.get('diagnostic_partial'):
            raise ValueError('Validación incompleta o diagnóstica')
        frame=pd.read_csv(pred)
        expected=dev.loc[dev.fold.eq(fold)].set_index('patient_id')
        if frame.patient_id.duplicated().any() or set(frame.patient_id)!=set(expected.index):
            raise ValueError('Pacientes de validación no coinciden con el fold oficial')
        if not np.array_equal(frame.label.to_numpy(), expected.loc[frame.patient_id,'pCR'].to_numpy()):
            raise ValueError('Etiquetas OOF no coinciden con desarrollo')
        if frame.n_slices.sum()!=member['val_slices'] or len(frame)!=member['val_patients']:
            raise ValueError('Cobertura incompleta de pacientes/cortes')
        observed=binary_metrics(frame.label,frame.probability,.5)
        if abs(observed['auc']-info['best_patient_auc'])>1e-12:
            raise ValueError('Las predicciones no corresponden al mejor checkpoint')
        frame['cv_fold']=fold; frames.append(frame[columns])
        sources.append(dict(fold=fold,patients=len(frame),best_epoch=info['best_epoch'],
            predictions=pred.relative_to(campaign).as_posix(),sha256=digest(pred),
            checkpoint_sha256=digest(run/'best_epoch.pt')))
    oof=pd.concat(frames,ignore_index=True).sort_values('patient_id').reset_index(drop=True)
    saved=list((campaign/'reports').glob('SUMMARY_*/oof_predictions_patient.csv'))
    if not saved:raise ValueError('No hay un informe OOF completo guardado')
    for path in saved:
        reference=pd.read_csv(path)[columns].sort_values('patient_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(oof,reference,check_exact=False,rtol=0,atol=1e-15)
    return oof, dict(campaign_id=campaign.name,campaign_sha256=digest(manifest),
        samples_sha256=digest(samples_csv),members=sources,
        saved_oof_reports=[dict(file=p.relative_to(campaign).as_posix(),sha256=digest(p)) for p in sorted(saved)])


def build(campaign, samples_csv=ROOT/'metadata/samples.csv', catalog_path=ROOT/'models/app_bundle/catalog.json'):
    oof,source=recover_oof(campaign,samples_csv)
    catalog=json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    model=catalog['modalities']['multimodal']
    if model['campaign_id']!=source['campaign_id'] or [m['fold'] for m in model['members']]!=list(range(5)) or any(
        a['checkpoint_sha256']!=b['original_sha256'] for a,b in zip(source['members'],model['members'])):
        raise ValueError('La recomendación no corresponde a los cinco modelos de la aplicación')
    report=analyze(oof.label,oof.probability)
    report['source']=source;report['model_checksum']=model['checksum']
    return report


def load_analysis(path=ROOT/'config/app_oof_threshold.json', model_checksum=None):
    path=Path(path)
    if not path.exists():return None
    report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('schema_version')!=1 or report.get('scope')!='multimodal_patient_oof_development_only'
            or report['methodology']['external_test_used'] is not False
            or (model_checksum is not None and report.get('model_checksum')!=model_checksum)
            or [r['threshold'] for r in report['thresholds']]!=list(GRID)):
        raise ValueError('Análisis de umbral incompatible con el modelo')
    best=report['recommended']; original=report['original']
    if best not in report['thresholds'] or original not in report['thresholds'] or original['threshold']!=.5:
        raise ValueError('Métricas de umbral incoherentes')
    if best['threshold']!=report['recommended_threshold'] or best['youden'] < max(r['youden'] for r in report['thresholds'])-1e-12:
        raise ValueError('Selección de Youden incoherente')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--output',default=str(ROOT/'config/app_oof_threshold.json'))
    args=parser.parse_args();report=build(args.campaign)
    # Nunca sobrescribir un análisis anterior, informe o checkpoint.
    with Path(args.output).open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(recommended=report['recommended'],n=report['n'],output=args.output),indent=2))


if __name__=='__main__':main()
