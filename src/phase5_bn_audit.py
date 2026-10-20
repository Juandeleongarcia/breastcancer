"""Audita BatchNorm con pesos Fase 4 congelados; recalibra solo con todo train."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.configuration import resolve_config,project_path
from src.dataset import BreastDCEDataset
from src.experiment import (set_seed,select_device,create_criterion,validate,
                            create_run_directory,environment_info,TrainingBudgetExceeded,make_datasets)
from src.metrics import save_evaluation_artifacts,analyze_probabilities
from src.model import BreastCancerCNN
from src.phase5_diagnostics import rank_info


def feature_info(embeddings,predictions):
    matrix=torch.cat(embeddings)
    info=rank_info(matrix)
    frame=pd.DataFrame(matrix.numpy())
    frame['patient_id']=predictions.patient_id.to_numpy()
    patients=frame.groupby('patient_id').mean()
    labels=predictions.groupby('patient_id').label.first().reindex(patients.index).to_numpy()
    values=patients.to_numpy()
    means=[values[labels==label].mean(0) for label in [0,1]]
    within=np.sqrt(np.mean(np.sum((values-np.stack(means)[labels])**2,axis=1)))
    distance=float(np.linalg.norm(means[1]-means[0]))
    info.update(patient_features=rank_info(torch.tensor(values)),
                class_centroid_distance=distance,within_class_rms_radius=float(within),
                centroid_distance_over_within_radius=distance/max(float(within),1e-12))
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    args=parser.parse_args();session=project_path(args.session)
    budget=json.loads((session/'budget.json').read_text(encoding='utf-8'))
    if budget['deadline_unix']-time.time()<720:
        print('Sin doce minutos de margen: auditoría completa BN pendiente, sin iniciarla.',flush=True);return
    source=project_path('models/RUN_20261020_184801_F4_NO_WEIGHT_PC_a59a7ae89dc0/checkpoints/best_epoch.pt')
    payload=torch.load(source,map_location='cpu',weights_only=True)
    cfg=resolve_config(payload['config']);cfg.update(DEVICE='cpu',DEADLINE_UNIX=budget['deadline_unix']-30,
                USE_AUGMENTATION=False,RUN_NAME='F5_DIAG_BN_RECALIB')
    set_seed(cfg['SEED']);device=select_device(cfg)
    train_ds,val_ds=make_datasets(cfg)
    assert train_ds.df.patient_id.nunique()==877 and val_ds.df.patient_id.nunique()==220
    assert not set(train_ds.df.patient_id)&set(val_ds.df.patient_id)
    model=BreastCancerCNN(cfg);model.load_state_dict(payload['model_state_dict'])
    parameters={n:p.detach().clone() for n,p in model.named_parameters()}
    criterion=create_criterion(cfg,device)
    val_loader=DataLoader(val_ds,batch_size=cfg['BATCH_SIZE'],shuffle=False,num_workers=0)
    embeddings=[]
    handle=model.features[-1].register_forward_hook(lambda module,inputs,output:embeddings.append(output.detach().flatten(1).cpu()))
    started=time.time();baseline=validate(model,val_loader,criterion,device,cfg)
    if 'val_patient_auc' in payload:
        assert abs(baseline['patient']['auc']-payload['val_patient_auc'])<1e-12
    baseline_features=feature_info(embeddings,baseline['slice_predictions']);embeddings.clear()
    frozen_train=validate(model,DataLoader(train_ds,batch_size=cfg['BATCH_SIZE'],shuffle=False,num_workers=0),criterion,device,cfg)
    embeddings.clear()
    frozen=dict(source=str(source),train=dict(loss=frozen_train['loss'],patient=frozen_train['patient'],slice=frozen_train['slice']),
                validation=dict(loss=baseline['loss'],patient=baseline['patient'],slice=baseline['slice']),
                learned_parameters_unchanged=True,interpretation='Mismos pesos en modo eval sobre todo train y toda validación; sin dropout, sin augmentation ni actualizaciones.')
    (session/'frozen_train_audit.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    if budget['deadline_unix']-time.time()<(time.time()-started)*1.10+30:
        handle.remove();print('Evaluación congelada conservada; sin margen para recalibrar BN.',flush=True);return
    model.eval()
    for layer in model.modules():
        if isinstance(layer,nn.BatchNorm2d):
            layer.reset_running_stats();layer.momentum=None;layer.train()
    train_loader=DataLoader(train_ds,batch_size=cfg['BATCH_SIZE'],shuffle=True,num_workers=0,
                            generator=torch.Generator().manual_seed(cfg['SEED']))
    # No gradientes ni dropout: únicamente se actualizan buffers de BatchNorm.
    seen=0
    with torch.no_grad():
        for batch in train_loader:
            if time.time()>=cfg['DEADLINE_UNIX']:raise TrainingBudgetExceeded('Deadline en auditoría BN')
            model(batch['image']);seen+=len(batch['label'])
            embeddings.clear()
    assert seen==8745
    recalibrated=validate(model,val_loader,criterion,device,cfg)
    recalibrated_features=feature_info(embeddings,recalibrated['slice_predictions']);handle.remove()
    assert all(torch.equal(parameters[n],p) for n,p in model.named_parameters())
    directory=create_run_directory(cfg);details=directory/'details'
    (details/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (details/'environment.json').write_text(json.dumps(environment_info(device),indent=2),encoding='utf-8')
    save_evaluation_artifacts(recalibrated,directory,details/'predictions')
    analyze_probabilities(recalibrated,directory,details)
    summary=dict(status='diagnostico_recalibracion_BN_sin_actualizar_pesos',run=directory.name,
        source=str(source),source_epoch=payload['epoch'],train_slices_for_BN=seen,train_patients=877,
        frozen_train=frozen['train'],
        validation_patients=220,learned_parameters_unchanged=True,seconds=time.time()-started,
        original=dict(loss=baseline['loss'],patient=baseline['patient'],slice=baseline['slice'],features=baseline_features),
        recalibrated=dict(loss=recalibrated['loss'],patient=recalibrated['patient'],slice=recalibrated['slice'],features=recalibrated_features),
        interpretation='Diagnóstico de buffers con todo train y validación completa; no es un nuevo entrenamiento de dos épocas ni un resultado de test.')
    for path in [directory/'summary.json',session/'bn_audit.json']:
        path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    rows=[]
    for stage,result in [('original',baseline),('BN_recalibrated',recalibrated)]:
        rows.append(dict(stage=stage,loss=result['loss'],**{f'patient_{k}':v for k,v in result['patient'].items() if k!='confusion_matrix'}))
    pd.DataFrame(rows).to_csv(directory/'history.csv',index=False)
    torch.save(dict(config=cfg,model_state_dict=model.state_dict(),diagnostic_only=True,resume_supported=False,
                    source_checkpoint=str(source),BN_train_slices=seen),directory/'checkpoints/bn_recalibrated.pt')
    (directory/'analysis.md').write_text('Auditoría de BatchNorm: mismos pesos aprendidos y mismos pacientes de validación.\n\nSolo se recalibran buffers con todos los cortes de train, dropout desactivado. No es un entrenamiento nuevo ni se usa test.\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
