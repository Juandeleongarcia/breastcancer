"""Validación cruzada oficial: train público únicamente, un modelo nuevo por fold."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.configuration import ROOT, project_path, load_config, apply_statistics
from src.cv_data import audit_samples, fold_frames, fit_fold_statistics, artifact_hash
from src.metrics import binary_metrics
from src.run_layout import technical_file


def portable(path):
    path=Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def save_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def implementation():
    files=['src/cross_validation.py','src/cv_data.py','src/model.py','src/dataset.py',
           'src/experiment.py','src/metrics.py','src/configuration.py','src/checkpointing.py',
           'src/augmentation.py','src/run_layout.py','src/batch_train.py','src/resume_validation.py',
           'src/layout_compatibility.json']
    return {name:artifact_hash(project_path(name)) for name in files}


def prepare_campaign(model='SPATIAL_SMALL', output_root='models', protocol=None, template=None, smoke=False):
    protocol=protocol or load_config('config/phase9/protocol.json')
    if model not in ['SPATIAL_SMALL','SPATIAL_CONTROL']:
        raise ValueError('Arquitectura CV desconocida')
    frame,audit=audit_samples(protocol['samples_csv'],protocol.get('patients_csv'),protocol['samples_sha256'])
    template=template or load_config(f'config/phase9/{model}.json')
    if (max(template['CONV_CHANNELS'])>128 or template['INPUT_CHANNELS']!=3
            or template['PRIMARY_METRIC']!='patient_auc' or not 1<=template['EPOCHS']<=20
            or not template['EVALUATE_TRAIN'] or template['CLASSIFICATION_THRESHOLD']!=.5
            or not template['USE_EARLY_STOPPING'] or template['PATIENCE']<1):
        raise ValueError('Configuración CV fuera del protocolo')
    stamp=datetime.now(ZoneInfo('Europe/Madrid')).strftime('%Y%m%d_%H%M%S')
    root=project_path(output_root)/f'CV9_{stamp}_{model}_{"SMOKE_" if smoke else ""}{uuid4().hex[:12]}'
    root.mkdir(parents=True,exist_ok=False)
    save_json(root/'template.json',template)
    state=dict(schema_version=1,model=model,diagnostic=bool(smoke),protocol=protocol,
               source_sha256=audit['source_sha256'],implementation=implementation(),
               template_sha256=artifact_hash(root/'template.json'),folds={},audit=audit)
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(root/'sources.zip','x',compression=ZIP_DEFLATED) as archive:
        for name in state['implementation']:
            archive.write(project_path(name),name)
    for fold in range(5):
        folder=root/f'fold_{fold}';folder.mkdir()
        tr,va=fold_frames(frame,fold,1 if smoke else None,template['SEED'])
        tr.to_csv(folder/'train.csv',index=False);va.to_csv(folder/'validation.csv',index=False)
        state['folds'][str(fold)]=dict(status='prepared',train_sha256=artifact_hash(folder/'train.csv'),
              validation_sha256=artifact_hash(folder/'validation.csv'),
              train_patients=tr.patient_id.nunique(),val_patients=va.patient_id.nunique(),
              train_slices=len(tr),val_slices=len(va))
    save_json(root/'audit.json',audit)
    pd.DataFrame(audit['folds']).to_csv(root/'fold_inventory.csv',index=False)
    save_json(root/'campaign.json',state)
    return root


def checked_campaign(root):
    root=project_path(root)
    state=load_config(root/'campaign.json')
    if state['implementation']!=implementation() or state['template_sha256']!=artifact_hash(root/'template.json'):
        raise ValueError('El ejecutor/configuración cambió: revisar o preparar otra campaña')
    protocol=state['protocol']
    frame,audit=audit_samples(protocol['samples_csv'],protocol.get('patients_csv'),state['source_sha256'])
    for key,entry in state['folds'].items():
        folder=root/f'fold_{key}'
        for name,field in [('train.csv','train_sha256'),('validation.csv','validation_sha256')]:
            if artifact_hash(folder/name)!=entry[field]:
                raise ValueError('CSV derivado alterado: '+str(folder/name))
        tr,va=fold_frames(frame,int(key),1 if state['diagnostic'] else None,load_config(root/'template.json')['SEED'])
        if not pd.read_csv(folder/'train.csv').equals(tr) or not pd.read_csv(folder/'validation.csv').equals(va):
            raise ValueError('Partición derivada distinta del fold oficial')
        for name,digest in entry.get('artifacts',{}).items():
            if artifact_hash(folder/name)!=digest:
                raise ValueError('Estadísticas/configuración derivada alterada: '+name)
    return root,state,frame


def fit_job(root, fold):
    root,state,frame=checked_campaign(root)
    entry=state['folds'][str(fold)];folder=root/f'fold_{fold}'
    if entry.get('config_file'):
        return load_config(folder/entry['config_file'])
    cfg=load_config(root/'template.json')
    cfg.update(TRAIN_CSV=portable(folder/'train.csv'),VAL_CSV=portable(folder/'validation.csv'),
               DATA_DIR=state['protocol']['data_dir'],CV_FOLD=fold,CV_SOURCE_SHA256=state['source_sha256'],
               RESULTS_DIR=portable(folder/'runs'),RUN_NAME=f'F9_{state["model"]}_FOLD{fold}'+('_SMOKE' if state['diagnostic'] else ''),
               EXPECTED_TRAIN_PATIENTS=entry['train_patients'],EXPECTED_VAL_PATIENTS=entry['val_patients'],
               SMOKE_TEST=state['diagnostic'],SMOKE_PATIENTS_PER_CLASS=1)
    if state['diagnostic']:
        cfg['EPOCHS']=2
    token=uuid4().hex[:12]
    stats_name=f'statistics_{token}.json';config_name=f'config_{token}.json'
    training,_=fold_frames(frame,fold,1 if state['diagnostic'] else None,cfg['SEED'])
    stats=fit_fold_statistics(cfg,training,folder/stats_name)
    cfg['STATS_PREPROCESSING']=stats['preprocessing']
    cfg=apply_statistics(cfg,portable(folder/stats_name))
    save_json(folder/config_name,cfg)
    entry.update(config_file=config_name,artifacts={name:artifact_hash(folder/name) for name in [stats_name,config_name]},status='ready')
    save_json(root/'campaign.json',state)
    return cfg


def aggregate_oof(expected, predictions):
    """Solo una predicción por paciente, de su fold excluido, y cobertura íntegra."""
    expected=expected[expected.split.eq('train')][['patient_id','pCR','fold']].drop_duplicates()
    if predictions.patient_id.duplicated().any() or set(predictions.patient_id)!=set(expected.patient_id):
        raise ValueError('OOF incompleto o pacientes repetidos/ajenos')
    joined=predictions.merge(expected,on='patient_id',validate='one_to_one')
    if not joined.fold.eq(joined.cv_fold).all() or not joined.label.eq(joined.pCR).all():
        raise ValueError('Una predicción OOF procede de otro fold/etiqueta')
    return binary_metrics(joined.label,joined.probability,.5)


def summarize_campaign(root):
    root,state,frame=checked_campaign(root)
    rows=[];predictions=[];completed=[]
    for fold in range(5):
        entry=state['folds'][str(fold)]
        row=dict(fold=fold,status=entry['status'],train_patients=entry['train_patients'],val_patients=entry['val_patients'])
        if entry.get('last_run') and entry['status']=='complete':
            directory=project_path(entry['last_run'])
            summary=load_config(technical_file(directory,'summary.json'))
            history=pd.read_csv(directory/'history.csv')
            best=history[history.epoch.eq(summary['best_epoch'])].iloc[0]
            metrics=summary['validation']['patient']
            row.update(run=entry['last_run'],best_epoch=summary['best_epoch'],epochs=summary['epochs_completed'],
                       val_loss=summary['validation']['loss'],train_eval_loss=best.train_eval_loss,
                       auc_patient=metrics['auc'],auc_slice=summary['validation']['slice']['auc'],
                       accuracy=metrics['accuracy'],balanced_accuracy=(metrics['recall']+metrics['specificity'])/2,
                       sensitivity=metrics['recall'],specificity=metrics['specificity'],f1=metrics['f1'])
            pred=pd.read_csv(technical_file(directory,'predictions_patient.csv'))
            pred['cv_fold']=fold
            # Comprobar cobertura incluso antes del resumen OOF completo.
            selected=pd.read_csv(root/f'fold_{fold}'/'validation.csv')
            aggregate_oof(selected,pred)
            predictions.append(pred)
            if not summary['diagnostic_partial'] and not state['diagnostic']:
                completed.append(metrics['auc'])
        rows.append(row)
    result=dict(diagnostic=state['diagnostic'],complete_folds=len(completed),required_folds=5,
                failed_folds=sum(e['status']=='error' for e in state['folds'].values()),
                auc_mean=float(np.mean(completed)) if completed else None,
                auc_std_sample=float(np.std(completed,ddof=1)) if len(completed)>1 else None,
                provisional=len(completed)!=5,oof_patient_auc=None,
                note='CV de desarrollo con checkpoint seleccionado en el mismo fold; test externo reservado')
    if len(completed)==5:
        pooled=pd.concat(predictions,ignore_index=True)
        result['oof_patient']=aggregate_oof(frame,pooled)
        result['oof_patient_auc']=result['oof_patient']['auc']
        # Snapshots únicos: una actualización no pisa un resumen anterior.
        token=uuid4().hex[:12]
        pooled.to_csv(root/f'oof_predictions_{token}.csv',index=False)
        from src.metrics import plot_roc_curve
        plot_roc_curve(pooled.label,pooled.probability,root/f'oof_roc_{token}.png',
                       title='ROC OOF paciente · CV de desarrollo, sin test externo')
    token=uuid4().hex[:12]
    pd.DataFrame(rows).to_csv(root/f'fold_comparison_{token}.csv',index=False)
    save_json(root/f'cv_summary_{token}.json',result)
    return result


def run_campaign(root, folds=None, device='cuda', resume=False):
    from src.batch_train import completed_run, interrupted_checkpoint
    from src.check_suite import require_rocm
    from src.experiment import train
    root,state,_=checked_campaign(root)
    if not state['diagnostic']:
        if device!='cuda':
            raise RuntimeError('Entrenamientos CV completos requieren GPU ROCm; no CPU')
        require_rocm()
    for fold in folds if folds is not None else range(5):
        try:
            cfg=fit_job(root,fold)
            root,state,_=checked_campaign(root)
            entry=state['folds'][str(fold)]
            if entry['status']=='complete':
                continue
            cfg['DEVICE']=device
            done=completed_run(cfg) if not state['diagnostic'] else None
            if done:
                entry.update(status='complete',last_run=portable(done))
                save_json(root/'campaign.json',state)
                continue
            existing=list(project_path(cfg['RESULTS_DIR']).glob('RUN_*'))
            checkpoint=interrupted_checkpoint(cfg) if resume else None
            if existing and checkpoint is None:
                raise ValueError('Hay una ejecución previa: usar --resume-interrupted o una campaña nueva; no reiniciar silenciosamente')
            entry['status']='running';save_json(root/'campaign.json',state)
            directory,summary=train(cfg,resume_from=checkpoint)
            entry.update(last_run=portable(directory),status='complete' if
                (summary.get('epochs_completed',0)>=cfg['EPOCHS'] or summary.get('stopped_early')) else 'interrupted')
            save_json(root/'campaign.json',state)
            summarize_campaign(root)
            if entry['status']=='interrupted':
                break
        except Exception as error:
            root,state,_=checked_campaign(root)
            state['folds'][str(fold)].update(status='error',error=str(error))
            save_json(root/'campaign.json',state)
            print(f'Fold {fold}: {error}',flush=True)
    return summarize_campaign(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['audit','prepare','run','summarize','smoke'])
    parser.add_argument('--model',choices=['SPATIAL_SMALL','SPATIAL_CONTROL'],default='SPATIAL_SMALL')
    parser.add_argument('--campaign')
    parser.add_argument('--folds',type=int,nargs='+',choices=range(5))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume-interrupted',action='store_true')
    parser.add_argument('--print-path',action='store_true',help='En prepare, imprimir solo la ruta para capturarla en una variable')
    args=parser.parse_args()
    if args.print_path and args.action!='prepare':
        parser.error('--print-path solo se usa con prepare')
    if args.action=='audit':
        protocol=load_config('config/phase9/protocol.json')
        _,report=audit_samples(protocol['samples_csv'],protocol['patients_csv'],protocol['samples_sha256'])
        print(json.dumps(report,indent=2));return
    if args.action in ['prepare','smoke']:
        root=prepare_campaign(args.model,smoke=args.action=='smoke')
        print(str(root) if args.print_path else 'Campaña: '+str(root),flush=True)
        if args.action=='prepare':return
        result=run_campaign(root,folds=[0],device='cpu')
    else:
        if not args.campaign:parser.error('Indicar --campaign')
        result=(run_campaign(args.campaign,args.folds,args.device,args.resume_interrupted)
                if args.action=='run' else summarize_campaign(args.campaign))
    print(json.dumps(result,indent=2))
    if result['failed_folds']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
