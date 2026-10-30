"""Exportación administrativa: inferencia y estadísticas públicas, nunca verdad terreno."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from app_services.results import read_campaign, CampaignReport, FoldResult
from src.configuration import ROOT, load_config, file_hash
from src.metrics import binary_metrics
from src.model import BreastCancerCNN
from src.phase13_model import SpatialSmallMultimodal
from src.run_layout import technical_file

CONFIG_KEYS=['INPUT_CHANNELS','IMAGE_SIZE','INPUT_MODE','USE_CLIPPING','CHANNEL_P01','CHANNEL_P99',
 'USE_STANDARDIZATION','CHANNEL_MEANS','CHANNEL_STDS','STATS_PROVENANCE','STATS_PREPROCESSING',
 'CONV_CHANNELS','CONVS_PER_BLOCK','SPATIAL_GRID','KERNEL_SIZE','PADDING','POOL_SIZE','USE_BATCH_NORM',
 'DROPOUT','CLASSIFICATION_THRESHOLD','BATCH_SIZE','SEED','LEARNING_RATE','WEIGHT_DECAY','OPTIMIZER',
 'USE_WEIGHTED_LOSS','EPOCHS','PATIENCE','SCHEDULER','SCHEDULER_FACTOR','SCHEDULER_PATIENCE','SCHEDULER_MIN_LR']


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def discover(models=ROOT/'models'):
    models=Path(models);found={'image':[],'multimodal':[],'external':[]}
    for outer in models.glob('CV9_*SPATIAL_SMALL*'):
        candidates=[outer,*outer.glob('CV9_*SPATIAL_SMALL*')]
        for root in candidates:
            if (root/'campaign.json').is_file() and not read_json(root/'campaign.json').get('diagnostic'):
                found['image'].append(root.resolve())
    for root in (models/'phase13').glob('CV13_*'):
        if (root/'campaign.json').is_file() and not read_json(root/'campaign.json').get('SMOKE_TEST'):
            found['multimodal'].append(root.resolve())
    for root in (models/'phase13/external_test').glob('ENSEMBLE13_*'):
        if (root/'evaluation_manifest.json').is_file() and read_json(root/'evaluation_manifest.json').get('status')=='complete':
            found['external'].append(root.resolve())
    for key in found:found[key]=list(dict.fromkeys(found[key]))
    return found


def select(found,choices=None):
    result={};choices=choices or {}
    for kind,values in found.items():
        requested=choices.get(kind)
        matches=[p for p in values if p.name==requested or str(p)==requested] if requested else values
        if len(matches)!=1:raise ValueError(f'{kind}: {len(matches)} campañas posibles; seleccionar explícitamente con --{kind}')
        result[kind]=matches[0]
    external=read_json(result['external']/'evaluation_manifest.json')
    source=str(external['campaign']).replace('\\','/').rstrip('/').split('/')[-1]
    if result['multimodal'].name!=source:raise ValueError('El test guardado pertenece a otra campaña multimodal')
    return result


def multimodal_report(root):
    from src.phase13 import checked,_fold_result
    root,state,cfg,dev,patients,external=checked(root)
    report=CampaignReport(root,'SPATIAL_SMALL_MULTIMODAL',cfg,[])
    for fold in range(5):
        if state['folds'][str(fold)]['status']!='complete':raise ValueError('CV13 incompleta')
        row,history,pred,metrics=_fold_result(root,state,cfg,dev,fold)
        run=root/state['folds'][str(fold)]['last_run']
        report.folds.append(FoldResult(fold,'Completo',True,run=run,history=history,predictions=pred,
            config=read_json(technical_file(run,'config.json')),summary=read_json(technical_file(run,'summary.json')),metrics=metrics))
    report.oof=pd.concat([f.predictions.assign(cv_fold=f.fold) for f in report.folds],ignore_index=True)
    if report.oof.patient_id.duplicated().any() or set(report.oof.patient_id)!=set(dev.patient_id):raise ValueError('OOF CV13 incompleto')
    report.oof_metrics=binary_metrics(report.oof.label,report.oof.probability,.5)
    return report


def roc_data(predictions):
    fpr,tpr,_=roc_curve(predictions.label,predictions.probability)
    return dict(fpr=fpr.tolist(),tpr=tpr.tolist())


def public_report(report,output,kind):
    if not report.complete:raise ValueError('Campaña no verificable: '+str([f.issue for f in report.folds]))
    rows=[]
    for fold in report.folds:
        numeric=['epoch','learning_rate','next_learning_rate','train_loss','train_eval_loss','val_loss']
        numeric += [name for name in fold.history if any(name.startswith(p) for p in ['train_eval_patient_','val_patient_'])
                    and name.rsplit('_',1)[-1] in ['auc','accuracy','recall','specificity','precision','f1']]
        history=fold.history[[name for name in dict.fromkeys(numeric) if name in fold.history]]
        history_file=f'{kind}_fold_{fold.fold}_history.csv';history.to_csv(output/history_file,index=False)
        rows.append(dict(fold=fold.fold,best_epoch=int(fold.summary['best_epoch']),epochs=len(fold.history),
            metrics=fold.metrics,roc=roc_data(fold.predictions),history=history_file))
    return dict(campaign_id=report.root.name,architecture=report.architecture,metrics=report.metrics,
        folds=rows,oof=dict(metrics=report.oof_metrics,roc=roc_data(report.oof)),
        development_patients=len(report.oof),scope='internal_cross_validation_not_external_test')


def build(output,models=ROOT/'models',choices=None):
    chosen=select(discover(models),choices);output=Path(output).resolve()
    if output.exists():raise ValueError('La carpeta de salida ya existe; usar otra versión para conservarla')
    for path in chosen.values():
        if output.is_relative_to(path):raise ValueError('No escribir dentro de una campaña histórica')
    image=read_campaign(chosen['image'],'SPATIAL_SMALL');multi=multimodal_report(chosen['multimodal'])
    external=read_json(chosen['external']/'metrics.json');manifest=read_json(chosen['external']/'evaluation_manifest.json')
    predictions=pd.read_csv(chosen['external']/'predictions_patient.csv')
    observed=binary_metrics(predictions.label,predictions.probability,.5)
    if len(predictions)!=176 or predictions.patient_id.duplicated().any():raise ValueError('Test guardado incompleto')
    for key,value in observed.items():
        saved=external['patient'][key]
        if isinstance(value,list):
            if saved!=value:raise ValueError('Matriz externa guardada incoherente')
        elif not np.isclose(saved,value,rtol=0,atol=1e-10):raise ValueError('Métrica externa guardada incoherente')
    if manifest['threshold']!=.5 or manifest['weights']!=[.2]*5:raise ValueError('Contrato externo no compatible')
    output.mkdir(parents=True)
    catalog=dict(schema_version=1,labels={'0':'no pCR · enfermedad residual','1':'pCR · respuesta patológica completa'},
        modalities={},external=dict(metrics=external['patient'],roc=roc_data(predictions),report_id=chosen['external'].name,
        scope='stored_external_evaluation_no_model_reevaluation',threshold=.5),
        data_license='CC BY-NC 4.0',training_performed=False)
    for kind,report in [('image',image),('multimodal',multi)]:
        result=public_report(report,output,kind);members=[]
        for fold in report.folds:
            raw=fold.run/'best_epoch.pt';payload=torch.load(raw,map_location='cpu',weights_only=True)
            if payload['epoch']!=fold.summary['best_epoch'] or payload.get('epoch_incomplete'):raise ValueError('Mejor checkpoint no coincide con el manifiesto')
            cfg={key:payload['config'][key] for key in CONFIG_KEYS}
            if kind=='multimodal':
                clinical=payload['clinical_transform']
                if raw.name!='best_epoch.pt' or file_hash(raw)!=manifest['members'][fold.fold]['checkpoint_sha256']:
                    raise ValueError('El informe externo no corresponde a los pesos multimodales elegidos')
                model=SpatialSmallMultimodal(cfg,len(clinical['columns']))
            else:clinical=None;model=BreastCancerCNN(cfg)
            model.load_state_dict(payload['model_state_dict'],strict=True)
            if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise ValueError('Pesos no finitos')
            filename=f'{kind}_fold_{fold.fold}.pt'
            clean=dict(schema='breastdcedl-inference-v1',kind=kind,config=cfg,clinical_transform=clinical,
                       model_state_dict=model.state_dict(),epoch=int(payload['epoch']))
            torch.save(clean,output/filename)
            members.append(dict(fold=fold.fold,file=filename,sha256=file_hash(output/filename),
                original_sha256=file_hash(raw),epoch=int(payload['epoch']),parameters=sum(p.numel() for p in model.parameters())))
        result['members']=members;result['checksum']=hashlib.sha256(''.join(m['sha256'] for m in members).encode()).hexdigest()
        catalog['modalities'][kind]=result
    # Un único ejemplo público de TRAIN, sin ID original, etiquetas ni clínica por paciente.
    frame=pd.read_csv(ROOT/'metadata/samples.csv');row=frame[frame.split.eq('train')].iloc[0]
    files=[]
    for phase,key in [('PRE','path_pre'),('EARLY','path_early'),('LATE','path_late')]:
        name=f'public_example_{phase}.png';shutil.copyfile(ROOT/'breastdcedl'/row[key],output/name);files.append(name)
    catalog['example']=dict(files=files,label='Ejemplo público de train · un corte',license='CC BY-NC 4.0',
        attribution='BreastDCEDL MinCrop; Fridman et al., Scientific Data 2026; I-SPY1/I-SPY2/Duke.',
        source='https://doi.org/10.5281/zenodo.18114231',warning='Ejemplo de funcionamiento; no mide generalización. No incluye etiqueta.')
    (output/'ATTRIBUTION.txt').write_text(catalog['example']['attribution']+'\n'+catalog['example']['source']+
        '\nDatos y ejemplo: CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/\n',encoding='utf-8')
    catalog['files']={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    save_json(output/'catalog.json',catalog)
    archive=output.with_suffix('.zip')
    if archive.exists():raise ValueError('El ZIP ya existe')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED) as package:
        for path in sorted(output.iterdir()):package.write(path,path.name)
    return dict(directory=str(output),archive=str(archive),sha256=file_hash(archive),bytes=archive.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--models',default=str(ROOT/'models'))
    for kind in ['image','multimodal','external']:parser.add_argument('--'+kind)
    args=parser.parse_args()
    result=build(args.output,args.models,{k:getattr(args,k) for k in ['image','multimodal','external'] if getattr(args,k)})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
