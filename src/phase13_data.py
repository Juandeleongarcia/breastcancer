"""Índice oficial y unión clínica por paciente; test reservado hasta el comando explícito."""
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.configuration import project_path, canonical_csv_hash
from src.dataset import BreastDCEDataset
from src.phase13_clinical import FEATURES, ClinicalTransform, selected_patients


def read_index(protocol):
    path = project_path(protocol['samples_csv'])
    if canonical_csv_hash(path) != protocol['samples_sha256']:
        raise ValueError('El índice oficial cambió; no se generan particiones nuevas')
    frame = pd.read_csv(path,dtype={'patient_id':str,'sample_id':str})
    keys = ['sample_id','patient_id','fold','split','path_pre','path_early','path_late','pCR']
    if any(k not in frame for k in keys) or frame[keys].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError('Índice incompleto o muestras duplicadas')
    if not frame.groupby('patient_id')[['fold','split']].nunique().eq(1).all().all():
        raise ValueError('Una paciente cambia de fold o split')
    if not frame.split.isin(['train','test']).all():
        raise ValueError('Split desconocido')
    dev = frame[frame.split.eq('train')].reset_index(drop=True)
    reserved = frame[frame.split.eq('test')]
    if (set(dev.fold.unique()) != set(range(5)) or not reserved.fold.eq(-1).all()
            or not dev.pCR.isin([0,1]).all() or not dev.groupby('patient_id').pCR.nunique().eq(1).all()):
        raise ValueError('Folds/etiquetas de desarrollo no válidos o test fuera de -1')
    if set(dev.patient_id) & set(reserved.patient_id):
        raise ValueError('Pacientes compartidos con test')
    validate_paths(dev)
    # No devolver etiquetas ni variables clínicas externas durante desarrollo.
    return dev, set(reserved.patient_id)


def validate_paths(frame):
    paths = ['path_pre','path_early','path_late']
    if frame[paths].stack().duplicated().any():
        raise ValueError('Rutas compartidas entre cortes')
    for row in frame.itertuples():
        stems = set()
        for key,phase in zip(paths,['PRE','EARLY','LATE']):
            text = getattr(row,key);path = PurePosixPath(text)
            if ('\\' in text or path.is_absolute() or '..' in path.parts
                    or path.parts[:2] != ('dataset',row.split) or path.parent.name != row.patient_id
                    or not path.name.endswith('_'+phase+'.png')):
                raise ValueError('Ruta incompatible con paciente/split/fase')
            stems.add(path.stem.rsplit('_',1)[0])
        if len(stems) != 1:
            raise ValueError('Fases de cortes diferentes')


def clinical_table(path, ids):
    # Lista blanca: pCR, split, test, dataset y otros campos jamás entran al tensor.
    patients = pd.read_csv(project_path(path),usecols=['pid',*FEATURES],dtype={'pid':str})
    selected_patients(patients, list(ids))
    return patients[patients.pid.isin(set(ids))].reset_index(drop=True)


def partition(frame, fold):
    if fold not in range(5) or not frame.fold.isin(range(5)).all() or not frame.split.eq('train').all():
        raise ValueError('Solo folds oficiales de desarrollo')
    tr = frame[frame.fold.ne(fold)].reset_index(drop=True)
    va = frame[frame.fold.eq(fold)].reset_index(drop=True)
    if set(tr.patient_id)&set(va.patient_id) or set(tr.pCR.unique())!={0,1} or set(va.pCR.unique())!={0,1}:
        raise ValueError('Particiones solapadas o sin ambas clases')
    return tr, va


def audit(protocol, check_images=False):
    dev, external_ids = read_index(protocol)
    patients = clinical_table(protocol['patients_csv'],dev.patient_id.unique())
    source = pd.read_csv(project_path(protocol['patients_csv']))
    selected = source[source.pid.isin(set(dev.patient_id))]
    merged = dev[['patient_id','pCR']].drop_duplicates().merge(selected[['pid','pCR']],
        left_on='patient_id',right_on='pid',suffixes=('_image','_clinical'),validate='one_to_one')
    if not merged.pCR_image.eq(merged.pCR_clinical).all():
        raise ValueError('Etiquetas clínicas e índice de imagen contradictorios en desarrollo')
    folds = []
    for fold in range(5):
        tr,va = partition(dev,fold)
        labels = va[['patient_id','pCR']].drop_duplicates().pCR
        folds.append(dict(fold=fold,train_patients=tr.patient_id.nunique(),train_slices=len(tr),
            val_patients=va.patient_id.nunique(),val_slices=len(va),
            no_pcr=int(labels.eq(0).sum()),pcr=int(labels.eq(1).sum())))
    missing_images = 0
    if check_images:
        missing_images = sum(not (project_path(protocol['data_dir'])/path).is_file()
                             for name in ['path_pre','path_early','path_late'] for path in dev[name])
        if missing_images:
            raise ValueError(f'Faltan {missing_images} imágenes de desarrollo')
    return dict(development_patients=dev.patient_id.nunique(),development_slices=len(dev),
        reserved_patients=len(external_ids),external_labels_analyzed=False,folds=folds,
        clinical_schema={name:str(kind) for name,kind in selected.dtypes.items()},
        clinical_missing_development=selected.isna().sum().astype(int).to_dict(),
        subtype_counts_development=patients.HR_HER2_STATUS.fillna('__MISSING__').value_counts().to_dict(),
        clinical_predictors=list(FEATURES),tum_vol='Excluded: GUIA.md defines volume but does not establish baseline timing',
        development_images_checked=check_images, missing_images=missing_images)


class MultimodalDataset(Dataset):
    def __init__(self, csv_path, data_dir, config, patients, transform, augment=False):
        self.images = BreastDCEDataset(csv_path,data_dir,config=config,augment=augment)
        self.df = self.images.df
        ids = self.df.patient_id.astype(str).unique().tolist()
        values = transform.transform(patients,ids)
        self.clinical = {pid:torch.from_numpy(value.copy()) for pid,value in zip(ids,values)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        row = self.images[index]
        row['clinical'] = self.clinical[str(row['patient_id'])].clone()
        return row
