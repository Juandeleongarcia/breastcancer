"""Contrato futuro autocontenido, sin conectar ni modificar Streamlit."""
import numpy as np
import pandas as pd
import torch

from src.dataset import BreastDCEDataset
from src.phase13_clinical import ClinicalTransform
from src.phase13_model import SpatialSmallMultimodal, ARCHITECTURE
from src.phase13_engine import inference_contract
from src.experiment import select_device


class ImageTransform:
    _build_channels=BreastDCEDataset._build_channels
    _clip_channels=BreastDCEDataset._clip_channels
    _standardize=BreastDCEDataset._standardize

    def __init__(self,config):
        self.config=config
        if config['STATS_PROVENANCE']!='verified_train_fit':raise ValueError('Faltan estadísticas train verificadas')
        if config['STATS_PREPROCESSING']!={k:config[k] for k in ['INPUT_MODE','USE_CLIPPING','CHANNEL_P01','CHANNEL_P99']}:
            raise ValueError('Representación y estadísticas incompatibles')
        for name in ['CHANNEL_P01','CHANNEL_P99','CHANNEL_MEANS','CHANNEL_STDS']:
            if np.asarray(config[name]).shape!=(3,) or not np.isfinite(config[name]).all():
                raise ValueError('Estadísticas de imagen no válidas')
        if min(config['CHANNEL_STDS'])<=0 or not (np.array(config['CHANNEL_P01'])<config['CHANNEL_P99']).all():
            raise ValueError('Escalas/clipping no válidos')

    def __call__(self,phases):
        phases=np.asarray(phases)
        size=self.config['IMAGE_SIZE']
        if phases.dtype!=np.uint8 or phases.shape!=(3,size,size):raise ValueError('Se requieren tres fases PNG8 compatibles')
        pre,early,late=torch.from_numpy(phases.copy()).float()/255.
        return self._standardize(self._clip_channels(self._build_channels(pre,early,late)))


def load_bundle(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    cfg=payload['config']
    if (payload.get('schema_version')!=13 or payload.get('architecture')!=ARCHITECTURE
            or cfg['SMOKE_TEST'] or cfg['TRAINING_SCOPE']!='final_development'
            or payload['epoch']!=cfg['EPOCHS']):
        raise ValueError('Inferencia final requiere un modelo final completo, no CV ni SMOKE_TEST')
    clinical=ClinicalTransform(payload['clinical_transform'])
    if payload['inference_contract']!=inference_contract(cfg,clinical.state):raise ValueError('Contrato de inferencia incoherente')
    cfg=dict(cfg,DEVICE=device);target=select_device(cfg)
    image_transform=ImageTransform(cfg)
    model=SpatialSmallMultimodal(cfg,clinical.dimension).to(target)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    return dict(model=model,config=cfg,clinical=clinical,images=image_transform,device=target,
                contract=payload['inference_contract'])


def predict_patient(bundle,phases,clinical_values,patient_id='inference'):
    if set(clinical_values)!={'age','HR_HER2_STATUS'}:
        raise ValueError('Aportar solo age y HR_HER2_STATUS; los ausentes se indican con None')
    array=np.asarray(phases)
    if array.ndim!=4 or not len(array):raise ValueError('Se requiere una pila no vacía de tripletas temporales')
    clinical=bundle['clinical'].transform(pd.DataFrame([dict(pid=patient_id,**clinical_values)]),[patient_id])
    probabilities=[]
    with torch.inference_mode():
        for offset in range(0,len(array),bundle['config']['BATCH_SIZE']):
            images=torch.stack([bundle['images'](p) for p in array[offset:offset+bundle['config']['BATCH_SIZE']]])
            values=torch.from_numpy(np.repeat(clinical,len(images),axis=0))
            logits=bundle['model'](images.to(bundle['device']),values.to(bundle['device'])).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    frame=pd.DataFrame(dict(patient_id=patient_id,slice_index=range(len(probabilities)),probability=probabilities))
    probability=float(frame.groupby('patient_id').probability.mean().iloc[0]);threshold=bundle['contract']['threshold']
    return dict(patient_probability=probability,no_pcr_probability=1-probability,threshold=threshold,
                prediction=int(probability>=threshold),slices=frame)
