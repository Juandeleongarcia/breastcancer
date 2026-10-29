"""Test externo único del ensemble F13: cinco mejores checkpoints CV, sin refit.

Módulo fuera de SOURCES de F13 deliberadamente: actualizarlo no invalida las
campañas/checkpoints que siguen entrenándose con la implementación original.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.configuration import project_path, load_config, canonical_csv_hash, file_hash
from src.experiment import select_device, environment_info
from src.phase13 import checked, _fold_result, training_contract
from src.phase13_clinical import ClinicalTransform, digest, feature_hash
from src.phase13_data import clinical_table, validate_paths
from src.phase13_engine import inference_contract
from src.phase13_inference import ImageTransform
from src.phase13_model import ARCHITECTURE, SpatialSmallMultimodal
from src.phase13_reports import unique_directory, write_json, outcome_plots

OUTPUT_ROOT='models/phase13/external_test'
EXPECTED_TEST_PATIENTS=176


@dataclass
class Member:
    fold: int
    checkpoint: Path
    sha256: str
    epoch: int
    config: dict
    clinical: dict
    contract: dict


def inside(root,value):
    path=(Path(root)/value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Un artefacto de campaña apunta fuera de su carpeta')
    return path


def checked_payload(member):
    if file_hash(member.checkpoint)!=member.sha256:
        raise ValueError('Checkpoint modificado después de fijar el ensemble')
    payload=torch.load(member.checkpoint,map_location='cpu',weights_only=True)
    if file_hash(member.checkpoint)!=member.sha256:
        raise ValueError('El checkpoint cambió durante la lectura')
    cfg=payload.get('config',{})
    if (payload.get('schema_version')!=13 or payload.get('architecture')!=ARCHITECTURE
            or payload.get('epoch_incomplete') or cfg.get('SMOKE_TEST')
            or cfg.get('TRAINING_SCOPE')!='cross_validation' or cfg.get('CV_FOLD')!=member.fold
            or payload.get('epoch')!=member.epoch or payload.get('best_epoch')!=member.epoch
            or payload.get('training_contract')!=member.contract
            or payload.get('clinical_transform')!=member.clinical):
        raise ValueError('Mejor checkpoint CV incompatible, parcial o de otro fold')
    runtime={'TRAIN_CSV','VAL_CSV','DATA_DIR','RESULTS_DIR','RUN_NAME','DEVICE','USE_GPU','STATS_PATH'}
    if {k:v for k,v in cfg.items() if k not in runtime}!=member.contract['configuration']:
        raise ValueError('Configuración del checkpoint distinta del perfil ajustado del fold')
    if payload.get('inference_contract')!=inference_contract(cfg,member.clinical):
        raise ValueError('Contrato de preprocesamiento/inferencia incoherente')
    if (cfg.get('CLASSIFICATION_THRESHOLD')!=.5 or cfg.get('IMAGE_SIZE')!=256
            or cfg.get('MAX_TRAIN_BATCHES') or cfg.get('MAX_VAL_BATCHES')
            or cfg.get('TRAIN_PATIENTS_PER_CLASS')):
        raise ValueError('Solo checkpoints completos 256x256 y umbral previamente fijado en 0,50')
    ImageTransform(cfg)
    ClinicalTransform(member.clinical)
    history=payload.get('history',[])
    if (len(history)!=member.epoch or history[-1]['epoch']!=member.epoch
            or not np.isclose(history[-1]['val_patient_auc'],payload.get('best_auc',float('nan')),atol=5e-6,rtol=0)):
        raise ValueError('Checkpoint y selección por AUC de validación no coinciden')
    model=SpatialSmallMultimodal(cfg,len(member.clinical['columns']))
    model.load_state_dict(payload['model_state_dict'],strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Pesos o buffers no finitos en el checkpoint; no iniciar test')
    return payload


def preflight(campaign,device='cpu',*,_expected_test_patients=EXPECTED_TEST_PATIENTS):
    """Solo integridad de desarrollo y pesos; no imágenes, clínica ni métricas externas."""
    root,state,base,dev,patients,external=checked(campaign)
    if (state['SMOKE_TEST'] or state['architecture']!=ARCHITECTURE
            or len(external)!=_expected_test_patients):
        raise ValueError('Se requiere CV F13 definitiva y el conjunto externo reservado completo')
    target=select_device(dict(base,DEVICE=device))
    members=[]
    for fold in range(5):
        entry=state['folds'][str(fold)]
        if entry['status']!='complete':raise ValueError(f'Fold {fold} no completo; esperar a que termine')
        result=_fold_result(root,state,base,dev,fold)
        if result is None:raise ValueError(f'Fold {fold} no verificable')
        row,history,predictions,metrics=result
        cfg=load_config(inside(root,entry['job']))
        clinical=load_config(inside(root,entry['clinical']))
        tr=root/f'fold_{fold}'/'train.csv';va=root/f'fold_{fold}'/'validation.csv'
        contract=training_contract(state,cfg,clinical,tr,va,patients)
        checkpoint=inside(root,entry['last_run'])/'best_epoch.pt'
        member=Member(fold,checkpoint,file_hash(checkpoint),int(row['best_epoch']),cfg,clinical,contract)
        payload=checked_payload(member)
        if not np.isclose(payload['best_auc'],row['auc'],atol=5e-6,rtol=0):
            raise ValueError('AUC del checkpoint distinto de la mejor validación guardada')
        members.append(member)
    if len({m.checkpoint.resolve() for m in members})!=5:
        raise ValueError('Deben ser cinco checkpoints independientes')
    return root,state,external,members,target


class ExternalDataset(Dataset):
    """Datos de entrada sin etiquetas; PRE/EARLY/LATE y clínica del mismo paciente."""
    def __init__(self,samples,patients,member,data_dir):
        columns=['sample_id','patient_id','path_pre','path_early','path_late']
        self.frame=samples[columns].reset_index(drop=True).copy()
        self.images=ImageTransform(member.config)
        self.root=project_path(data_dir)
        transform=ClinicalTransform(member.clinical)
        ids=self.frame.patient_id.unique().tolist()
        self.clinical=dict(zip(ids,transform.transform(patients,ids)))

    def __len__(self):return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index];phases=[]
        for key in ['path_pre','path_early','path_late']:
            with Image.open(self.root/row[key]) as image:
                if image.mode!='L' or image.size!=(256,256) or image.format!='PNG':
                    raise ValueError('El estudio externo no cumple el contrato PNG8 gris 256x256')
                phases.append(np.asarray(image).copy())
        return dict(image=self.images(np.stack(phases)),
            clinical=torch.from_numpy(self.clinical[row.patient_id].copy()),
            patient_id=row.patient_id,sample_id=row.sample_id)


def predict_member(member,samples,patients,data_dir,device):
    payload=checked_payload(member)
    model=SpatialSmallMultimodal(payload['config'],len(member.clinical['columns'])).to(device)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    dataset=ExternalDataset(samples,patients,member,data_dir)
    loader=DataLoader(dataset,batch_size=member.config['BATCH_SIZE'],shuffle=False,
        num_workers=member.config['NUM_WORKERS'],pin_memory=device.type=='cuda',
        generator=torch.Generator().manual_seed(member.config['SEED']))
    rows=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['image'].to(device),batch['clinical'].to(device)).squeeze(1)
            if not torch.isfinite(logits).all():raise FloatingPointError('Logits externos no finitos')
            rows.extend(dict(sample_id=str(sid),patient_id=str(pid),probability=float(p))
                for sid,pid,p in zip(batch['sample_id'],batch['patient_id'],torch.sigmoid(logits).cpu().tolist()))
    frame=pd.DataFrame(rows)
    if (len(frame)!=len(samples) or frame.sample_id.duplicated().any()
            or set(frame.sample_id)!=set(samples.sample_id)):
        raise ValueError('Inferencia externa parcial o con cortes duplicados')
    patient=frame.groupby('patient_id',sort=True).agg(probability=('probability','mean'),n_slices=('sample_id','size')).reset_index()
    return patient,frame


def average_patients(predictions,samples):
    if set(predictions)!=set(range(5)):raise ValueError('El ensemble exige exactamente cinco folds')
    result=samples.groupby('patient_id',sort=True).agg(n_slices=('sample_id','size')).reset_index()
    for fold in range(5):
        frame=predictions[fold]
        if (frame.patient_id.duplicated().any() or set(frame.patient_id)!=set(result.patient_id)
                or not np.isfinite(frame.probability).all() or not frame.probability.between(0,1).all()):
            raise ValueError('Predicciones por paciente incompletas, duplicadas o no válidas')
        joined=result[['patient_id','n_slices']].merge(frame,on='patient_id',validate='one_to_one',suffixes=('','_model'))
        if not joined.n_slices.eq(joined.n_slices_model).all():raise ValueError('Número de cortes inconsistente entre modelos')
        result=result.merge(frame[['patient_id','probability']].rename(columns={'probability':f'probability_fold_{fold}'}),
                            on='patient_id',validate='one_to_one')
    columns=[f'probability_fold_{fold}' for fold in range(5)]
    result['probability']=result[columns].mean(axis=1)
    return result


def external_samples(protocol,external_ids):
    if canonical_csv_hash(protocol['samples_csv'])!=protocol['samples_sha256']:
        raise ValueError('Índice oficial modificado después del preflight')
    frame=pd.read_csv(project_path(protocol['samples_csv']),dtype={'patient_id':str,'sample_id':str})
    test=frame[frame.split.eq('test') & frame.fold.eq(-1)].reset_index(drop=True)
    if (set(test.patient_id)!=external_ids or not test.pCR.isin([0,1]).all()
            or not test.groupby('patient_id').pCR.nunique().eq(1).all() or test.sample_id.duplicated().any()):
        raise ValueError('Cobertura/etiquetas del test externo no válidas')
    validate_paths(test)
    return test


def evaluate_ensemble(campaign,device='cpu',confirm=False,*,_output_root=None,
                      _expected_test_patients=EXPECTED_TEST_PATIENTS):
    if not confirm:raise ValueError('Indicar --confirm-external-test para iniciar la evaluación única')
    root,state,external,members,target=preflight(campaign,device,_expected_test_patients=_expected_test_patients)
    output=project_path(_output_root or OUTPUT_ROOT).resolve()
    if output.is_relative_to(root.resolve()) or output.is_relative_to(project_path(state['protocol']['data_dir']).resolve()):
        raise ValueError('Los resultados deben quedar fuera de la campaña y del dataset originales')
    # Identidad de campaña, independiente de pesos/umbral: cambiarlos no habilita otro intento.
    identity=digest(dict(campaign=root.name,configuration=state['configuration_sha256'],
                         source=state['protocol']['samples_sha256']))
    ledger_dir=output/'_evaluations';ledger_dir.mkdir(parents=True,exist_ok=True)
    ledger=ledger_dir/f'{identity}.json'
    selection=dict(schema_version=1,evaluation_type='external_cv_checkpoint_ensemble',status='reserved',
        campaign=str(root.resolve()),campaign_identity=identity,expected_patients=len(external),
        threshold=.5,threshold_policy='fixed_before_test_no_tuning',weights=[.2]*5,
        aggregation='mean_over_5_models_of_mean_slice_sigmoid_probabilities_per_patient',
        training_performed=False,source_sha256=file_hash('src/phase13_ensemble.py'),
        frozen_training_implementation=state['implementation'],
        members=[dict(fold=m.fold,checkpoint=str(m.checkpoint),checkpoint_sha256=m.sha256,
            selected_epoch=m.epoch,clinical_transform=m.clinical,
            image_preprocessing=inference_contract(m.config,m.clinical)['image_preprocessing']) for m in members])
    try:
        with ledger.open('x',encoding='utf-8') as stream:json.dump(selection,stream,indent=2)
    except FileExistsError as error:
        raise ValueError('Esta campaña ya tiene una evaluación externa reservada/iniciada/completada; no se repite') from error
    report=None
    try:
        report=unique_directory(output,'ENSEMBLE13')
        selection.update(status='started',report=str(report))
        write_json(ledger,selection);write_json(report/'evaluation_manifest.json',selection)
        write_json(report/'environment.json',environment_info(target))
        # Las etiquetas entran después de congelar los cinco checkpoints, pesos y umbral.
        samples=external_samples(state['protocol'],external)
        patients=clinical_table(state['protocol']['patients_csv'],samples.patient_id.unique())
        selection['external_clinical_sha256']=feature_hash(patients,samples.patient_id.unique())
        technical=report/'internals';technical.mkdir()
        predictions={}
        inputs=samples.drop(columns=['pCR'])
        for member in members:
            print(f'Ensemble: fold {member.fold}, checkpoint de época {member.epoch}; inferencia sin entrenamiento',flush=True)
            patient,slices=predict_member(member,inputs,patients,state['protocol']['data_dir'],target)
            predictions[member.fold]=patient
            slices.to_csv(technical/f'predictions_slice_fold_{member.fold}.csv',index=False)
        pooled=average_patients(predictions,inputs)
        truth=samples.groupby('patient_id',sort=True).agg(label=('pCR','first')).reset_index()
        pooled=pooled.merge(truth,on='patient_id',validate='one_to_one')
        pooled['prediction']=(pooled.probability>=.5).astype(int)
        if len(pooled)!=len(external):raise ValueError('Faltan pacientes en el ensemble')
        pooled.to_csv(report/'predictions_patient.csv',index=False)
        metrics=outcome_plots(pooled,report,.5,'Test externo · ensemble de cinco modelos CV · paciente')
        result=dict(evaluation_type=selection['evaluation_type'],n_models=5,n_patients=len(pooled),
            threshold=.5,patient=metrics,selection_used_test=False,training_performed=False,
            note='Rendimiento externo del ensemble CV; no de un modelo único reentrenado con los 1097 pacientes.')
        write_json(report/'metrics.json',result)
        pd.DataFrame([dict(n_models=5,threshold=.5,**{k:v for k,v in metrics.items() if k!='confusion_matrix'})]).to_csv(report/'metrics.csv',index=False)
        selection['status']='complete';write_json(report/'evaluation_manifest.json',selection);write_json(ledger,selection)
        return report,result
    except BaseException as error:
        selection.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        write_json(ledger,selection)
        if report:write_json(report/'evaluation_manifest.json',selection)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--confirm-external-test',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Verificar los cinco mejores checkpoints sin ejecutar inferencia externa')
    args=parser.parse_args()
    if args.check_only and args.confirm_external_test:parser.error('Elegir --check-only o --confirm-external-test')
    try:
        if args.check_only:
            root,state,external,members,target=preflight(args.campaign,args.device)
            print(json.dumps(dict(campaign=str(root),device=str(target),ready=True,external_test_executed=False,
                folds=[dict(fold=m.fold,best_epoch=m.epoch,checkpoint_sha256=m.sha256) for m in members]),indent=2))
        else:
            report,result=evaluate_ensemble(args.campaign,args.device,args.confirm_external_test)
            print(json.dumps(dict(report=str(report),**result),indent=2))
    except KeyboardInterrupt:parser.exit(130,'Ensemble interrumpido; reserva conservada para revisión, sin repetición automática.\n')
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:parser.exit(1,f'Ensemble F13: {error}\n')


if __name__=='__main__':main()
