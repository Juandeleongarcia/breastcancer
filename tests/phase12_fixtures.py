"""Fixtures exclusivamente sintéticas en temporales; nunca son resultados clínicos."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.configuration import load_config,canonical_csv_hash
from src.cross_validation import prepare_campaign,save_json,portable
from src.metrics import binary_metrics
from app_services.architectures import create_model


def png_bytes(array):
    stream=BytesIO();Image.fromarray(array).save(stream,format='PNG');return stream.getvalue()


def study_files(n=3,patient='TEST_ONLY'):
    grid=np.indices((256,256)).sum(axis=0)%120
    return [(f'{patient}_z{z:03d}_{phase}.png',png_bytes((grid+z+i*12).astype(np.uint8)))
            for z in range(n) for i,phase in enumerate(['PRE','EARLY','LATE'])]


def fitted_config(kind='SPATIAL_SMALL'):
    cfg=load_config('config/phase9/SPATIAL_SMALL.json' if kind=='SPATIAL_SMALL' else 'config/phase11/F11_CNN4.json')
    cfg.update(CHANNEL_P01=[0.,-.2,-.2],CHANNEL_P99=[1.,.7,.7],CHANNEL_MEANS=[.2,.1,.1],
               CHANNEL_STDS=[.15,.12,.12],STATS_PROVENANCE='verified_train_fit',
               TRAIN_CSV='fixture_train.csv',VAL_CSV='fixture_validation.csv')
    cfg['STATS_PREPROCESSING']={k:cfg[k] for k in ['INPUT_MODE','USE_CLIPPING','CHANNEL_P01','CHANNEL_P99']}
    return cfg


def checkpoint_fixture(root,kind='SPATIAL_SMALL'):
    cfg=fitted_config(kind)
    _,model=create_model(cfg)
    # Pesos constantes para contrastar mecánicamente sigmoid(0)=0.5. No entrenar.
    with torch.no_grad():
        model.classifier[-1].weight.zero_();model.classifier[-1].bias.zero_()
    filename=Path(root)/(kind+'.pt')
    torch.save(dict(config=cfg,model_state_dict=model.state_dict(),epoch=3,
                    test_fixture_only=True,epoch_incomplete=False),filename)
    return filename,cfg


def campaign_fixture(root,completed=5,kind='SPATIAL_SMALL'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=[]
    for fold in [-1,0,1,2,3,4]:
        split='test' if fold==-1 else 'train'
        for label in [0,1]:
            pid=f'TEST_F{fold+1}_{label}'
            for z in range(2):
                paths={column:f'dataset/{split}/{pid}/{pid}_z{z:03d}_{phase}.png'
                       for column,phase in [('path_pre','PRE'),('path_early','EARLY'),('path_late','LATE')]}
                rows.append(dict(patient_id=pid,sample_id=f'{pid}_z{z:03d}',pCR=label,split=split,fold=fold,**paths))
    frame=pd.DataFrame(rows);source=root/'samples.csv';frame.to_csv(source,index=False)
    cfg=fitted_config(kind);cfg.update(EPOCHS=3)
    protocol=dict(samples_csv=str(source),samples_sha256=canonical_csv_hash(source),data_dir=str(root))
    campaign=prepare_campaign(output_root=root/'campaigns',protocol=protocol,template=cfg)
    state=json.loads((campaign/'campaign.json').read_text())
    state['model']=kind
    for fold in range(completed):
        run=campaign/f'fold_{fold}/runs/RUN_fixture_{fold}'
        (run/'internals').mkdir(parents=True)
        actual=dict(cfg,TRAIN_CSV=str(campaign/f'fold_{fold}/train.csv'),VAL_CSV=str(campaign/f'fold_{fold}/validation.csv'))
        save_json(run/'internals/config.json',actual)
        history=pd.DataFrame(dict(epoch=[1,2,3],train_loss=[.6,.5,.4],train_eval_loss=[.65,.60,.55],
            val_loss=[.66,.69,.74],train_eval_patient_accuracy=[.5,.7,.9],val_patient_accuracy=[.5,.5,1.],
            train_eval_patient_auc=[.6,.8,.99],val_patient_auc=[.5,.75,1.],learning_rate=[.0005,.0005,.00025],diagnostic_partial=False))
        history.to_csv(run/'history.csv',index=False)
        va=pd.read_csv(actual['VAL_CSV'])
        patients=va[['patient_id','pCR']].drop_duplicates()
        probabilities=np.where(patients.pCR,.9,.1)
        predictions=pd.DataFrame(dict(patient_id=patients.patient_id,label=patients.pCR,probability=probabilities,n_slices=2))
        predictions.to_csv(run/'internals/predictions_patient.csv',index=False)
        metrics=binary_metrics(patients.pCR,probabilities)
        save_json(run/'internals/summary.json',dict(best_epoch=3,epochs_completed=3,requested_epochs=3,
            stopped_early=False,validation_complete=True,status='entrenamiento_completo',diagnostic_partial=False,
            validation=dict(loss=.74,patient=metrics,slice=metrics)))
        state['folds'][str(fold)].update(status='complete',last_run=portable(run))
    save_json(campaign/'campaign.json',state)
    return campaign
