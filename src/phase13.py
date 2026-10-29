"""Fase 13 independiente: SPATIAL_SMALL multimodal, CV, recuperación y test explícito."""
import argparse
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import signal
import sys

import numpy as np
import pandas as pd
import torch

from src.configuration import project_path, load_config, resolve_config, canonical_csv_hash, apply_statistics, file_hash
from src.cross_validation import portable
from src.cv_data import artifact_hash, fit_fold_statistics
from src.phase13_clinical import ClinicalTransform, feature_hash, digest
from src.phase13_data import read_index, clinical_table, partition, audit
from src.phase13_model import ARCHITECTURE, architecture_report
from src.phase13_engine import train_job
from src.phase13_reports import (unique_directory, write_json, history_plots, outcome_plots,
    mean_history, mean_plots, recommend, validate_patient_predictions)
from src.metrics import binary_metrics
from src.run_layout import technical_file

CONFIG='config/phase13/SPATIAL_SMALL_MULTIMODAL.json'
SOURCES=['src/phase13.py','src/phase13_model.py','src/phase13_clinical.py','src/phase13_data.py',
    'src/phase13_engine.py','src/phase13_reports.py','src/phase13_inference.py',CONFIG,
    'src/model.py','src/dataset.py','src/configuration.py','src/cv_data.py','src/checkpointing.py',
    'src/experiment.py','src/metrics.py','src/augmentation.py','src/cross_validation.py','src/run_layout.py',
    'config/config.py','config/phase9/SPATIAL_SMALL.json','config/phase9/protocol.json']


@contextmanager
def campaign_lock(campaign):
    root=project_path(campaign)
    if not (root/'campaign.json').is_file():raise ValueError('No existe la campaña F13')
    # El SO libera el bloqueo incluso al morir el proceso; el archivo puede permanecer.
    stream=(root/'execution.lock').open('a+b');acquired=False
    try:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if sys.platform=='win32':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError as error:
            raise RuntimeError('Otra ejecución utiliza esta campaña; no iniciar procesos simultáneos') from error
        yield
    finally:
        if acquired:
            stream.seek(0)
            if sys.platform=='win32':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def exclusive_campaign(function):
    @wraps(function)
    def wrapped(campaign,*args,**kwargs):
        with campaign_lock(campaign):return function(campaign,*args,**kwargs)
    return wrapped


def implementation():
    return {name:artifact_hash(project_path(name)) for name in SOURCES}


def template():
    return resolve_config(load_config(CONFIG))


def validate_config(cfg, smoke=False):
    original=load_config('config/phase9/SPATIAL_SMALL.json')
    keys=['CONV_CHANNELS','CONVS_PER_BLOCK','KERNEL_SIZE','PADDING','POOL_SIZE','INPUT_CHANNELS',
          'SPATIAL_GRID','USE_BATCH_NORM','DROPOUT','BATCH_SIZE','OPTIMIZER','LEARNING_RATE',
          'WEIGHT_DECAY','SCHEDULER','SCHEDULER_FACTOR','SCHEDULER_PATIENCE','SCHEDULER_MIN_LR',
          'SEED','USE_WEIGHTED_LOSS','USE_AUGMENTATION','INPUT_MODE','USE_CLIPPING','USE_STANDARDIZATION',
          'CLASSIFICATION_THRESHOLD','EVALUATE_TRAIN','PRIMARY_METRIC','USE_EARLY_STOPPING','PATIENCE']
    technical=bool(cfg.get('TECHNICAL_FIXTURE'))
    if technical and not smoke:raise ValueError('Fixture técnica solo en SMOKE_TEST')
    if technical:keys.remove('BATCH_SIZE')
    if (any(cfg.get(k)!=original[k] for k in keys) or cfg['ARCHITECTURE']!=ARCHITECTURE
            or cfg['CLINICAL_FEATURES']!=['HR_HER2_STATUS','age'] or cfg['CLINICAL_TUM_VOL']
            or not cfg['CLINICAL_AGE_MISSING_INDICATOR'] or not 1<=cfg['EPOCHS']<=20
            or cfg['NUM_WORKERS']<0 or (cfg['IMAGE_SIZE']!=256 and not technical)
            or cfg['MAX_TRAIN_BATCHES'] or cfg['MAX_VAL_BATCHES'] or cfg.get('TRAIN_PATIENTS_PER_CLASS')):
        raise ValueError('F13 requiere extractor/hiperparámetros originales y todos los pacientes/cortes')
    bounds=cfg['LOSS_PLOT_YLIM']
    if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
        raise ValueError('Límites verticales de loss no válidos')


def prepare(output_root='models/phase13', epochs=None, workers=None, smoke=False, protocol=None, config=None):
    cfg=dict(config or template())
    if epochs is not None:cfg['EPOCHS']=epochs
    if workers is not None:cfg['NUM_WORKERS']=workers
    cfg.update(SMOKE_TEST=bool(smoke),TRAINING_SCOPE='cross_validation')
    validate_config(cfg,smoke)
    protocol=protocol or load_config('config/phase9/protocol.json')
    dev,external=read_index(protocol)
    patients=clinical_table(protocol['patients_csv'],dev.patient_id.unique())
    report=audit(protocol)
    root=unique_directory(project_path(output_root),'CV13_'+ARCHITECTURE+('_SMOKE_TEST' if smoke else ''))
    write_json(root/'config.json',cfg)
    state=dict(schema_version=13,architecture=ARCHITECTURE,SMOKE_TEST=bool(smoke),protocol=protocol,
        implementation=implementation(),configuration_sha256=artifact_hash(root/'config.json'),
        development_clinical_sha256=feature_hash(patients,dev.patient_id.unique()),folds={},audit=report)
    for fold in range(5):
        folder=root/f'fold_{fold}';folder.mkdir()
        tr,va=partition(dev,fold)
        tr.to_csv(folder/'train.csv',index=False);va.to_csv(folder/'validation.csv',index=False)
        state['folds'][str(fold)]=dict(status='prepared',train_sha256=canonical_csv_hash(folder/'train.csv'),
            val_sha256=canonical_csv_hash(folder/'validation.csv'),train_patients=tr.patient_id.nunique(),
            val_patients=va.patient_id.nunique(),train_slices=len(tr),val_slices=len(va))
    write_json(root/'audit.json',report)
    pd.DataFrame(report['folds']).to_csv(root/'fold_inventory.csv',index=False)
    write_json(root/'campaign.json',state)
    return root


def checked(campaign):
    root=project_path(campaign);state=load_config(root/'campaign.json');cfg=load_config(root/'config.json')
    if (state.get('schema_version')!=13 or state['implementation']!=implementation()
            or state['configuration_sha256']!=artifact_hash(root/'config.json')):
        raise ValueError('Fuentes/configuración cambiadas: no continuar una campaña incompatible')
    validate_config(cfg,state['SMOKE_TEST'])
    dev,external=read_index(state['protocol'])
    patients=clinical_table(state['protocol']['patients_csv'],dev.patient_id.unique())
    if feature_hash(patients,dev.patient_id.unique())!=state['development_clinical_sha256']:
        raise ValueError('Las variables clínicas de desarrollo cambiaron')
    for fold in range(5):
        folder=root/f'fold_{fold}';entry=state['folds'][str(fold)];tr,va=partition(dev,fold)
        for name,expected,key in [('train.csv',tr,'train_sha256'),('validation.csv',va,'val_sha256')]:
            if (canonical_csv_hash(folder/name)!=entry[key]
                    or not pd.read_csv(folder/name).equals(expected)):
                raise ValueError('Partición derivada alterada')
        for name,sha in entry.get('artifacts',{}).items():
            if artifact_hash(root/name)!=sha:raise ValueError('Perfil clínico/imagen/configuración alterado')
        for name,sha in entry.get('result_artifacts',{}).items():
            if artifact_hash(root/name)!=sha:raise ValueError('Resultado completado alterado')
    return root,state,cfg,dev,patients,external


def fit_job(campaign,fold):
    root,state,base,dev,patients,external=checked(campaign)
    entry=state['folds'][str(fold)];folder=root/f'fold_{fold}'
    if entry.get('job'):
        cfg=load_config(root/entry['job'])
        clinical=load_config(root/entry['clinical'])
    else:
        tr,va=partition(dev,fold)
        clinical=ClinicalTransform.fit(patients,tr.patient_id.unique(),va.patient_id.unique(),external).state
        fit=unique_directory(folder/'internals','FIT')
        cfg=dict(base,TRAIN_CSV=portable(folder/'train.csv'),VAL_CSV=portable(folder/'validation.csv'),
            DATA_DIR=state['protocol']['data_dir'],CV_FOLD=fold,RUN_NAME=f'F13_FOLD{fold}',
            RESULTS_DIR=portable(folder/'runs'))
        stats=fit_fold_statistics(cfg,tr,fit/'image_statistics.json')
        cfg['STATS_PREPROCESSING']=stats['preprocessing']
        cfg=apply_statistics(cfg,portable(fit/'image_statistics.json'))
        write_json(fit/'clinical_transform.json',clinical);write_json(fit/'config.json',cfg)
        relative=lambda p:p.relative_to(root).as_posix()
        entry.update(job=relative(fit/'config.json'),clinical=relative(fit/'clinical_transform.json'),
            status='ready',artifacts={relative(p):artifact_hash(p) for p in fit.iterdir() if p.is_file()})
        write_json(root/'campaign.json',state)
    # Las rutas de CSV pueden remapearse al copiar la campaña; no se cambian parámetros científicos.
    cfg.update(TRAIN_CSV=portable(folder/'train.csv'),VAL_CSV=portable(folder/'validation.csv'))
    return cfg,clinical


def training_contract(state,cfg,clinical,train_csv,val_csv,patients):
    runtime={'TRAIN_CSV','VAL_CSV','DATA_DIR','RESULTS_DIR','RUN_NAME','DEVICE','USE_GPU','STATS_PATH'}
    scientific={key:value for key,value in cfg.items() if key not in runtime}
    train=pd.read_csv(project_path(train_csv))
    return dict(schema_version=13,implementation=state['implementation'],configuration=scientific,
        official_samples_sha256=state['protocol']['samples_sha256'],
        train_csv_sha256=canonical_csv_hash(train_csv),val_csv_sha256=canonical_csv_hash(val_csv) if val_csv else None,
        clinical_transform=clinical,train_clinical_sha256=feature_hash(patients,train.patient_id.unique()),
        development_clinical_sha256=state['development_clinical_sha256'])


def _run_path(root,entry):
    return root/entry['last_run'] if entry.get('last_run') else None


def _fold_result(root,state,cfg,dev,fold):
    entry=state['folds'][str(fold)];run=_run_path(root,entry)
    if run is None or not (run/'internals/summary.json').exists():return None
    summary=load_config(run/'internals/summary.json')
    if summary['status']!='complete':return None
    history=pd.read_csv(run/'history.csv')
    if (history.empty or history.epoch.tolist()!=list(range(1,len(history)+1))
            or len(history)!=summary['epochs_completed'] or not (run/'best_epoch.pt').is_file()
            or summary['SMOKE_TEST']!=state['SMOKE_TEST'] or not summary['validation_complete']
            or bool(summary.get('diagnostic_partial'))!=state['SMOKE_TEST']
            or not np.isfinite(history[['train_eval_loss','val_loss','val_patient_auc','learning_rate']]).all().all()):
        raise ValueError('Historial/checkpoint/resumen incompleto')
    if len(history)<cfg['EPOCHS'] and not (summary['stopped_early'] and
            len(history)-summary['best_epoch']>=cfg['PATIENCE']):
        raise ValueError('El fold no terminó épocas ni early stopping')
    best=history.loc[history.epoch.eq(summary['best_epoch'])].iloc[0]
    if int(history.loc[history.val_patient_auc.idxmax(),'epoch'])!=summary['best_epoch']:
        raise ValueError('Mejor época incorrecta o empate no reproducible')
    validation=partition(dev,fold)[1]
    prediction=pd.read_csv(run/'internals/predictions_patient.csv')
    validate_patient_predictions(prediction,validation)
    metric=binary_metrics(prediction.label,prediction.probability,cfg['CLASSIFICATION_THRESHOLD'])
    if not np.isclose(metric['auc'],best.val_patient_auc,atol=5e-6,rtol=0):
        raise ValueError('Predicciones e historial no corresponden al mejor AUC')
    row=dict(fold=fold,epochs=len(history),best_epoch=summary['best_epoch'],
        train_eval_loss=float(best.train_eval_loss),val_loss=float(best.val_loss),auc_slice=float(best.val_slice_auc),
        **{key:metric[key] for key in ['auc','accuracy','balanced_accuracy','precision','recall','specificity','f1']})
    return row,history,prediction,metric


def summarize(campaign):
    root,state,cfg,dev,patients,external=checked(campaign)
    rows=[];histories={};predictions=[]
    for fold in range(5):
        result=_fold_result(root,state,cfg,dev,fold)
        if result:
            row,history,prediction,metric=result
            if not state['SMOKE_TEST']:
                rows.append(row);histories[fold]=history
                predictions.append(prediction.assign(cv_fold=fold))
    result=dict(schema_version=13,architecture=ARCHITECTURE,SMOKE_TEST=state['SMOKE_TEST'],
        complete_folds=len(rows),required_folds=5,provisional=len(rows)!=5,
        excluded_diagnostics=bool(state['SMOKE_TEST']),auc_mean=float(np.mean([r['auc'] for r in rows])) if rows else None,
        auc_std_sample=float(np.std([r['auc'] for r in rows],ddof=1)) if len(rows)>1 else None,
        oof_patient_auc=None,folds=rows,external_test_used=False,
        note='Desarrollo: selección del checkpoint en la misma validación del fold; CV no anidada, no test externo')
    result['fold_metric_means']={key:dict(mean=float(np.mean([r[key] for r in rows])),
        std_sample=float(np.std([r[key] for r in rows],ddof=1)) if len(rows)>1 else None)
        for key in ['accuracy','balanced_accuracy','precision','recall','specificity','f1'] if rows}
    report=unique_directory(root/'reports','SUMMARY')
    pd.DataFrame(rows).to_csv(report/'fold_metrics.csv',index=False)
    means=mean_history(histories)
    if not means.empty:means.to_csv(report/'mean_history.csv',index=False);mean_plots(means,report)
    if rows:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.5));ax.bar([r['fold'] for r in rows],[r['auc'] for r in rows],color='#007E87')
        ax.set(xticks=range(5),ylim=(0,1),xlabel='Fold',ylabel='ROC AUC por paciente',title='Discriminación por fold · desarrollo')
        fig.tight_layout();fig.savefig(report/'fold_auc.png',dpi=160);plt.close(fig)
    if len(rows)==5:
        pooled=pd.concat(predictions,ignore_index=True)
        validate_patient_predictions(pooled,dev)
        expected=dev[['patient_id','fold']].drop_duplicates()
        if not pooled.merge(expected,on='patient_id',validate='one_to_one').cv_fold.eq(
                pooled.merge(expected,on='patient_id',validate='one_to_one').fold).all():
            raise ValueError('OOF de un fold no excluido del entrenamiento')
        pooled.to_csv(report/'oof_predictions_patient.csv',index=False)
        metrics=outcome_plots(pooled,report,cfg['CLASSIFICATION_THRESHOLD'],'OOF por paciente · cinco folds internos')
        result.update(oof_patient=metrics,oof_patient_auc=metrics['auc'])
        recommendation=recommend(rows,histories,cfg)
        target=root/'recommended_final_training.json'
        if target.exists() and load_config(target)!=recommendation:
            raise ValueError('Existe una recomendación diferente; no sobrescribir una selección previa')
        if not target.exists():write_json(target,recommendation)
        result['recommended_epochs']=recommendation['recommended_epochs']
    write_json(report/'summary.json',result)
    write_json(root/'latest_report.json',dict(path=report.relative_to(root).as_posix()))
    return result


def recovery_checkpoint(run):
    candidates=list(Path(run).glob('epoch_[0-9][0-9][0-9].pt'))
    recovery=Path(run)/'internals/recovery_epoch_start.pt'
    if recovery.exists():candidates.append(recovery)
    if not candidates:raise ValueError('No hay un checkpoint compatible para recuperar; crear una campaña nueva')
    def epoch(path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema_version')!=13 or not payload.get('resume_supported'):
            raise ValueError('Checkpoint de recuperación no compatible')
        return payload['epoch']
    return max(candidates,key=epoch)


@exclusive_campaign
def run(campaign, folds=None, device='auto', resume=False, allow_device_change=False):
    if folds is not None and (not folds or len(set(folds))!=len(folds) or any(f not in range(5) for f in folds)):
        raise ValueError('Indicar folds únicos de 0 a 4')
    for fold in folds if folds is not None else range(5):
        root,state,base,dev,patients,external=checked(campaign)
        entry=state['folds'][str(fold)]
        if _fold_result(root,state,base,dev,fold):
            if entry['status']!='complete':
                entry['status']='complete';write_json(root/'campaign.json',state)
            print(f'Fold {fold} completo: se omite',flush=True);continue
        cfg,clinical=fit_job(root,fold)
        root,state,base,dev,patients,external=checked(root);entry=state['folds'][str(fold)]
        previous=_run_path(root,entry)
        if previous and not resume:
            raise ValueError('Hay una ejecución previa: usar --resume-interrupted; no se reinicia desde cero')
        checkpoint=recovery_checkpoint(previous) if previous else None
        cfg['DEVICE']=device
        contract=training_contract(state,cfg,clinical,cfg['TRAIN_CSV'],cfg['VAL_CSV'],patients)
        from src.experiment import select_device
        from src.phase13_engine import compatible_checkpoint
        selected_device=select_device(cfg)
        if checkpoint:compatible_checkpoint(checkpoint,contract,cfg,selected_device,allow_device_change)
        directory=unique_directory(root/f'fold_{fold}'/'runs',f'RUN_F13_FOLD{fold}'+('_SMOKE_TEST' if state['SMOKE_TEST'] else '')+('_RESUME' if previous else ''))
        entry.update(status='running',last_run=directory.relative_to(root).as_posix())
        write_json(root/'campaign.json',state)
        try:
            summary=train_job(cfg,clinical,patients,contract,directory,checkpoint,allow_device_change)
            entry['status']=summary['status']
            if summary['status']=='complete':
                names=['history.csv','internals/summary.json','internals/predictions_patient.csv','internals/predictions_slice.csv']
                entry['result_artifacts']={str((directory/name).relative_to(root).as_posix()):artifact_hash(directory/name) for name in names}
            write_json(root/'campaign.json',state)
            summarize(root)
            if summary['status']!='complete':break
        except (Exception,KeyboardInterrupt) as error:
            entry.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error))
            write_json(root/'campaign.json',state)
            raise
    return summarize(campaign)


def compare(campaign,baseline_campaign):
    root,state,cfg,dev,patients,external=checked(campaign)
    current=summarize(root)
    if current['complete_folds']!=5:raise ValueError('Comparación requiere cinco folds F13 definitivos')
    baseline=project_path(baseline_campaign);old=load_config(baseline/'campaign.json')
    basecfg=load_config(baseline/'template.json')
    if (old.get('diagnostic') or old.get('model')!='SPATIAL_SMALL'
            or old['protocol']['samples_sha256']!=state['protocol']['samples_sha256']):
        raise ValueError('Se requiere una campaña real SPATIAL_SMALL con los mismos folds oficiales')
    rows=[];pooled=[]
    for fold in range(5):
        entry=old['folds'][str(fold)]
        if entry['status']!='complete':raise ValueError('Baseline con folds incompletos')
        for name,expected in zip(['train.csv','validation.csv'],partition(dev,fold)):
            if not pd.read_csv(baseline/f'fold_{fold}'/name).equals(expected):
                raise ValueError('Las particiones baseline difieren de los folds oficiales')
        path=project_path(entry['last_run'])
        if not path.is_dir():
            basename=str(entry['last_run']).replace('\\','/').rstrip('/').split('/')[-1]
            path=baseline/f'fold_{fold}'/'runs'/basename
        summary=load_config(technical_file(path,'summary.json'));history=pd.read_csv(path/'history.csv')
        if (summary.get('diagnostic_partial') or not summary.get('validation_complete')
                or len(history)!=summary['epochs_completed'] or (len(history)<basecfg['EPOCHS']
                and not (summary.get('stopped_early') and len(history)-summary['best_epoch']>=basecfg['PATIENCE']))):
            raise ValueError('Baseline diagnóstico o validación parcial')
        prediction=pd.read_csv(technical_file(path,'predictions_patient.csv'))
        validate_patient_predictions(prediction,partition(dev,fold)[1])
        metrics=binary_metrics(prediction.label,prediction.probability,basecfg['CLASSIFICATION_THRESHOLD'])
        best=history.loc[history.epoch.eq(summary['best_epoch'])].iloc[0]
        if not np.isclose(metrics['auc'],best.val_patient_auc,atol=5e-6,rtol=0):
            raise ValueError('Baseline: historial y predicciones incoherentes')
        rows.append(dict(fold=fold,model='SPATIAL_SMALL',**{k:metrics[k] for k in ['auc','accuracy','balanced_accuracy','recall','specificity','f1']},
            train_eval_loss=float(best.train_eval_loss),val_loss=float(best.val_loss)))
        pooled.append(prediction)
    oldpred=pd.concat(pooled,ignore_index=True);validate_patient_predictions(oldpred,dev)
    oldmetrics=binary_metrics(oldpred.label,oldpred.probability,basecfg['CLASSIFICATION_THRESHOLD'])
    report=unique_directory(root/'reports','COMPARISON')
    table=pd.DataFrame(rows+[dict(r,model=ARCHITECTURE) for r in current['folds']])
    table.to_csv(report/'paired_folds.csv',index=False)
    base_mean=float(np.mean([r['auc'] for r in rows]))
    differences={k:dict(baseline=basecfg.get(k),multimodal=cfg.get(k)) for k in basecfg
        if k not in ['TRAIN_CSV','VAL_CSV','DATA_DIR','RESULTS_DIR','RUN_NAME','DEVICE','USE_GPU','STATS_PATH',
                     'CHANNEL_P01','CHANNEL_P99','CHANNEL_MEANS','CHANNEL_STDS','STATS_PREPROCESSING','STATS_PROVENANCE'] and basecfg.get(k)!=cfg.get(k)}
    result=dict(baseline=dict(auc_mean=base_mean,auc_std_sample=float(np.std([r['auc'] for r in rows],ddof=1)),oof=oldmetrics),
        multimodal=current,auc_mean_difference=current['auc_mean']-base_mean,configuration_differences=differences,
        complexity=architecture_report(cfg,len(load_config(root/state['folds']['0']['clinical'])['columns'])),
        interpretation='Comparación descriptiva pareada; cinco folds no son muestras independientes. No prueba superioridad ni utiliza test.',
        baseline_checkpoint_required=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for label,frame in table.groupby('model'):ax.plot(frame.fold,frame.auc,'o-',label=label)
    ax.set(xticks=range(5),ylim=(0,1),xlabel='Fold',ylabel='ROC AUC por paciente',title='Comparación descriptiva · mismos folds internos')
    ax.legend();fig.tight_layout();fig.savefig(report/'paired_auc.png',dpi=160);plt.close(fig)
    write_json(report/'comparison.json',result)
    return result


@exclusive_campaign
def final_train(campaign, device='auto', selected=False, final_run=None, resume=False, allow_device_change=False):
    if not selected:raise ValueError('Entrenamiento final requiere --confirm-selection tras elegir la versión definitiva')
    root,state,base,dev,patients,external=checked(campaign)
    result=summarize(root)
    if result['complete_folds']!=5 or state['SMOKE_TEST']:raise ValueError('Se requieren cinco folds definitivos')
    recommendation=load_config(root/'recommended_final_training.json')
    if final_run:
        target=project_path(final_run);manifest=load_config(target/'final_training.json')
        if (manifest['recommendation_sha256']!=digest(recommendation) or manifest['implementation']!=implementation()
                or manifest['external_test_status']!='pending'):
            raise ValueError('El entrenamiento final o la selección cambió, o el test ya se inició')
        if manifest['status']=='complete':print('Modelo final ya completo; se omite');return target
        if not resume:raise ValueError('Usar --resume-interrupted para continuar el entrenamiento final')
        cfg=load_config(target/'config.json');clinical=load_config(target/'clinical_transform.json')
        if canonical_csv_hash(target/'development.csv')!=manifest['development_sha256']:
            raise ValueError('Desarrollo final alterado')
        stats=load_config(target/'image_statistics.json')
        if digest(dict(config=cfg,clinical=clinical,statistics=stats))!=manifest['profile_sha256']:
            raise ValueError('Perfil final alterado')
        previous=target/manifest['last_run'];checkpoint=recovery_checkpoint(previous)
    else:
        target=unique_directory(root/'final_models','FINAL13_'+ARCHITECTURE)
        dev.to_csv(target/'development.csv',index=False)
        cfg=dict(base,EPOCHS=recommendation['recommended_epochs'],TRAINING_SCOPE='final_development',CV_FOLD=-2,
            TRAIN_CSV=portable(target/'development.csv'),VAL_CSV=None,DATA_DIR=state['protocol']['data_dir'])
        clinical=ClinicalTransform.fit(patients,dev.patient_id.unique(),external_ids=external).state
        stats=fit_fold_statistics(cfg,dev,target/'image_statistics.json')
        cfg['STATS_PREPROCESSING']=stats['preprocessing'];cfg=apply_statistics(cfg,portable(target/'image_statistics.json'))
        write_json(target/'config.json',cfg);write_json(target/'clinical_transform.json',clinical)
        manifest=dict(schema_version=13,architecture=ARCHITECTURE,status='prepared',selection_confirmed=True,
            recommendation_sha256=digest(recommendation),recommendation=recommendation,implementation=implementation(),
            source_campaign=portable(root),protocol=state['protocol'],development_sha256=canonical_csv_hash(target/'development.csv'),
            profile_sha256=digest(dict(config=cfg,clinical=clinical,statistics=stats)),external_test_status='pending',
            threshold=cfg['CLASSIFICATION_THRESHOLD'],threshold_policy='fixed_before_external_test_no_tuning')
        checkpoint=None
    cfg.update(DEVICE=device,TRAIN_CSV=portable(target/'development.csv'))
    contract=training_contract(state,cfg,clinical,cfg['TRAIN_CSV'],None,patients)
    contract['final_lr_schedule']=recommendation['final_lr_schedule']
    directory=unique_directory(target/'runs','RUN_F13_FINAL'+('_RESUME' if checkpoint else ''))
    manifest.update(status='running',last_run=directory.relative_to(target).as_posix())
    write_json(target/'final_training.json',manifest)
    try:
        summary=train_job(cfg,clinical,patients,contract,directory,checkpoint,allow_device_change,
                          recommendation['final_lr_schedule'])
        manifest['status']=summary['status']
        if summary['status']=='complete':
            manifest['checkpoint']=(directory/'final_epoch.pt').relative_to(target).as_posix()
            manifest['checkpoint_sha256']=file_hash(directory/'final_epoch.pt')
        write_json(target/'final_training.json',manifest)
    except (Exception,KeyboardInterrupt) as error:
        manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error))
        write_json(target/'final_training.json',manifest);raise
    return target


def external_test(final_run,confirm=False,device='cpu'):
    if not confirm:raise ValueError('Test externo bloqueado: requiere --confirm-external-test y selección final previa')
    target=project_path(final_run);manifest=load_config(target/'final_training.json')
    if (manifest['status']!='complete' or not manifest['selection_confirmed']
            or manifest['external_test_status']!='pending' or manifest['implementation']!=implementation()):
        raise ValueError('Modelo final no disponible, incompatible o test ya iniciado')
    checkpoint=target/manifest['checkpoint']
    if file_hash(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint final alterado')
    # Reserva exclusiva antes de acceder a etiquetas/imágenes externas. No se reevalúa silenciosamente.
    with (target/'external_evaluation.lock').open('x',encoding='utf-8') as stream:stream.write('external test started\n')
    manifest['external_test_status']='started';write_json(target/'final_training.json',manifest)
    from src.phase13_inference import load_bundle
    from src.phase13_data import MultimodalDataset, validate_paths
    from src.phase13_engine import evaluate
    from src.experiment import create_criterion,select_device
    bundle=load_bundle(checkpoint,device)
    protocol=manifest['protocol']
    if canonical_csv_hash(protocol['samples_csv'])!=protocol['samples_sha256']:raise ValueError('Índice externo alterado')
    frame=pd.read_csv(project_path(protocol['samples_csv']))
    test=frame[frame.split.eq('test') & frame.fold.eq(-1)].reset_index(drop=True)
    dev=frame[frame.split.eq('train')]
    if test.empty or set(test.patient_id)&set(dev.patient_id) or not test.pCR.isin([0,1]).all():
        raise ValueError('Test reservado no válido')
    validate_paths(test)
    report=unique_directory(target/'external_test','TEST13')
    test.to_csv(report/'test_index.csv',index=False)
    patients=clinical_table(protocol['patients_csv'],test.patient_id.unique())
    cfg=dict(bundle['config'],DEVICE=str(bundle['device']),DATA_DIR=protocol['data_dir'])
    dataset=MultimodalDataset(report/'test_index.csv',protocol['data_dir'],cfg,patients,bundle['clinical'])
    result=evaluate(bundle['model'],dataset,create_criterion(cfg,bundle['device']),bundle['device'],cfg,
                    torch.Generator().manual_seed(cfg['SEED']))
    result['patient_predictions'].to_csv(report/'predictions_patient.csv',index=False)
    result['slice_predictions'].to_csv(report/'predictions_slice.csv',index=False)
    outcome_plots(result['patient_predictions'],report,cfg['CLASSIFICATION_THRESHOLD'],'Test externo reservado · paciente')
    write_json(report/'metrics.json',dict(patient=result['patient'],slice=result['slice'],loss=result['loss'],
        threshold=cfg['CLASSIFICATION_THRESHOLD'],checkpoint_sha256=manifest['checkpoint_sha256'],
        immutable_selection=manifest['recommendation_sha256']))
    manifest.update(external_test_status='complete',external_report=report.relative_to(target).as_posix())
    write_json(target/'final_training.json',manifest)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    for name in ['check','audit']:
        sub=subs.add_parser(name);sub.add_argument('--check-images',action='store_true')
        if name=='check':sub.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    for name in ['prepare','smoke']:
        sub=subs.add_parser(name);sub.add_argument('--epochs',type=int,default=1 if name=='smoke' else None)
        sub.add_argument('--workers',type=int,default=None);sub.add_argument('--output-root',default='models/phase13')
        if name=='prepare':sub.add_argument('--print-path',action='store_true')
        else:sub.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    for name in ['run','status','summarize','recommend','plots','compare','final-train']:
        sub=subs.add_parser(name);sub.add_argument('--campaign',required=True)
        if name in ['run','final-train']:
            sub.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
            sub.add_argument('--resume-interrupted',action='store_true');sub.add_argument('--allow-device-change',action='store_true')
        if name=='run':sub.add_argument('--folds',nargs='+',type=int,choices=range(5))
        if name=='plots':
            sub.add_argument('--loss-ymin',type=float,default=0);sub.add_argument('--loss-ymax',type=float,default=1)
        if name=='compare':sub.add_argument('--baseline-campaign',required=True)
        if name=='final-train':
            sub.add_argument('--confirm-selection',action='store_true');sub.add_argument('--final-run')
    sub=subs.add_parser('external-test');sub.add_argument('--final-run',required=True)
    sub.add_argument('--confirm-external-test',action='store_true');sub.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    args=parser.parse_args()
    # SIGTERM usa la misma recuperación de inicio de época que Ctrl+C.
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        if args.action in ['check','audit']:
            protocol=load_config('config/phase9/protocol.json');result=audit(protocol,args.check_images)
            if args.action=='check':
                from src.experiment import select_device
                from src.phase13_model import SpatialSmallMultimodal
                cfg=template();cfg['DEVICE']=args.device;device=select_device(cfg)
                dev,external=read_index(protocol);patients=clinical_table(protocol['patients_csv'],dev.patient_id.unique())
                tr,va=partition(dev,0);clinical=ClinicalTransform.fit(patients,tr.patient_id.unique(),va.patient_id.unique(),external)
                model=SpatialSmallMultimodal(cfg,clinical.dimension).to(device).eval()
                with torch.inference_mode():assert tuple(model(torch.zeros(1,3,256,256,device=device),torch.zeros(1,clinical.dimension,device=device)).shape)==(1,1)
                result.update(architecture=architecture_report(cfg,clinical.dimension),device=str(device),
                    torch=str(torch.__version__),hip=getattr(torch.version,'hip',None),cuda_available=torch.cuda.is_available())
        elif args.action in ['prepare','smoke']:
            root=prepare(args.output_root,args.epochs,args.workers,args.action=='smoke')
            if args.action=='prepare':print(str(root) if args.print_path else 'Campaña: '+str(root));return
            print('SMOKE_TEST completo por datos, excluido de CV definitiva:',root,flush=True)
            result=run(root,[0],args.device)
        elif args.action=='run':result=run(args.campaign,args.folds,args.device,args.resume_interrupted,args.allow_device_change)
        elif args.action=='status':
            root,state,*_=checked(args.campaign)
            result=dict(campaign=str(root),SMOKE_TEST=state['SMOKE_TEST'],folds=state['folds'])
        elif args.action in ['summarize','recommend']:
            result=summarize(args.campaign)
            if args.action=='recommend':
                if result['complete_folds']!=5:raise ValueError('Recomendación pendiente de cinco folds definitivos')
                result=load_config(project_path(args.campaign)/'recommended_final_training.json')
        elif args.action=='compare':result=compare(args.campaign,args.baseline_campaign)
        elif args.action=='plots':
            if not np.isfinite([args.loss_ymin,args.loss_ymax]).all() or args.loss_ymin>=args.loss_ymax:
                raise ValueError('Límites de loss inválidos')
            root,state,cfg,dev,*_=checked(args.campaign);report=unique_directory(root/'reports','PLOTS')
            for fold in range(5):
                path=_run_path(root,state['folds'][str(fold)])
                if path and (path/'history.csv').exists():
                    history_plots(pd.read_csv(path/'history.csv'),report/f'fold_{fold}',(args.loss_ymin,args.loss_ymax))
            result=dict(plots=str(report),loss_ylim=[args.loss_ymin,args.loss_ymax],original_data_unchanged=True)
        elif args.action=='final-train':result=dict(final_run=str(final_train(args.campaign,args.device,args.confirm_selection,
            args.final_run,args.resume_interrupted,args.allow_device_change)))
        else:result=dict(external_report=str(external_test(args.final_run,args.confirm_external_test,args.device)))
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except KeyboardInterrupt:
        print('Interrumpido. Reanudar con --resume-interrupted: se repite únicamente la época incompleta.',file=sys.stderr);raise SystemExit(130)
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
        parser.exit(1,f'F13: {error}\n')


if __name__=='__main__':main()
