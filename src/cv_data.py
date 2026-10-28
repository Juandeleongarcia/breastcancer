"""Folds oficiales, CSV derivados y estadísticas exclusivas del train de cada fold."""
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

from src.configuration import project_path, canonical_csv_hash


def audit_samples(samples_csv, patients_csv=None, expected_sha256=None):
    frame = pd.read_csv(project_path(samples_csv))
    required = ['sample_id','patient_id','split','fold','pCR','path_pre','path_early','path_late']
    if any(k not in frame for k in required) or frame[required].isna().any().any():
        raise ValueError('Índice incompleto o valores ausentes')
    digest = canonical_csv_hash(samples_csv)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('El índice oficial cambió; no se recalculan folds automáticamente')
    if frame.sample_id.duplicated().any() or not frame.pCR.isin([0,1]).all():
        raise ValueError('Muestras duplicadas o etiquetas no binarias')
    if not frame.split.isin(['train','test']).all():
        raise ValueError('Split público desconocido')
    if not frame.groupby('patient_id')[['split','fold','pCR']].nunique().eq(1).all().all():
        raise ValueError('Una paciente cambia de split/fold/etiqueta')
    train = frame[frame.split.eq('train')]
    test = frame[frame.split.eq('test')]
    if set(train.fold.unique()) != set(range(5)) or not test.fold.eq(-1).all():
        raise ValueError('Se requieren folds 0–4 exclusivamente en train y -1 en test')
    if frame[required[5:]].stack().duplicated().any():
        raise ValueError('Rutas de imagen duplicadas')
    for row in frame.itertuples():
        paths = [PurePosixPath(getattr(row,k)) for k in required[5:]]
        if any('\\' in str(p) or p.is_absolute() or '..' in p.parts for p in paths):
            raise ValueError('Ruta de imagen no portable o fuera del dataset')
        if any(p.parts[:2] != ('dataset',row.split) or p.parent.name != str(row.patient_id) for p in paths):
            raise ValueError('Ruta incompatible con el paciente/split público')
        if (len({p.stem.rsplit('_',1)[0] for p in paths}) != 1 or
                any(not p.name.endswith('_'+phase+'.png') for p,phase in zip(paths,['PRE','EARLY','LATE']))):
            raise ValueError('Fases de distinto paciente/corte')
    if patients_csv:
        patients = pd.read_csv(project_path(patients_csv))
        if patients.pid.duplicated().any():
            raise ValueError('Pacientes duplicados en patients.csv')
        lookup = patients.set_index('pid')
        for patient in frame[['patient_id','pCR','split']].drop_duplicates().itertuples():
            if patient.patient_id not in lookup.index:
                raise ValueError('Paciente ausente de patients.csv')
            source = lookup.loc[patient.patient_id]
            if int(source.pCR) != patient.pCR or source.split != patient.split or int(source.test) != int(patient.split=='test'):
                raise ValueError('Contradicción entre samples.csv y patients.csv')
    folds = []
    for fold in range(5):
        validation = train[train.fold.eq(fold)]
        training = train[~train.fold.eq(fold)]
        labels = validation[['patient_id','pCR']].drop_duplicates().pCR
        if set(labels) != {0,1}:
            raise ValueError('Fold sin ambas clases')
        folds.append(dict(fold=fold,train_patients=training.patient_id.nunique(),train_slices=len(training),
                          val_patients=len(labels),val_slices=len(validation),
                          val_patient_no_pcr=int(labels.eq(0).sum()),val_patient_pcr=int(labels.eq(1).sum()),
                          val_slice_no_pcr=int(validation.pCR.eq(0).sum()),val_slice_pcr=int(validation.pCR.eq(1).sum())))
    reserved = test[['patient_id','pCR']].drop_duplicates()
    report = dict(source_csv=str(samples_csv),source_sha256=digest,folds=folds,
                  train_patients=train.patient_id.nunique(),train_slices=len(train),
                  reserved_test_patients=len(reserved),reserved_test_slices=len(test),
                  reserved_test_no_pcr=int(reserved.pCR.eq(0).sum()),reserved_test_pcr=int(reserved.pCR.eq(1).sum()))
    return frame,report


def fold_frames(frame, fold, smoke_per_class=None, seed=42):
    if fold not in range(5):
        raise ValueError('Fold inválido')
    public_train = frame[frame.split.eq('train')]
    train = public_train[~public_train.fold.eq(fold)].copy()
    val = public_train[public_train.fold.eq(fold)].copy()
    if smoke_per_class:
        from src.experiment import select_patients
        train = select_patients(train,smoke_per_class,seed)
        val = select_patients(val,smoke_per_class,seed)
    if set(train.patient_id)&set(val.patient_id) or set(train.patient_id)&set(frame[frame.split.eq('test')].patient_id):
        raise ValueError('Fuga de pacientes')
    return train.reset_index(drop=True),val.reset_index(drop=True)


def _quantile(hist, q):
    rank = (int(hist.sum())-1)*q
    lo, hi = int(np.floor(rank)),int(np.ceil(rank))
    cdf = hist.cumsum()
    a = np.searchsorted(cdf,lo,side='right')-255
    b = np.searchsorted(cdf,hi,side='right')-255
    return float((a+(b-a)*(rank-lo))/255.)


def fit_fold_statistics(cfg, training, output):
    """Una pasada exacta de histogramas PNG8/diferencias, sin muestreo ni validación.

    511 bins por canal cubren [-255,255]. Momentos float64 tras clipping;
    redondeo de la resta float32 del loader puede diferir en la última cifra.
    """
    frame = pd.read_csv(project_path(cfg['TRAIN_CSV']))
    if set(frame.sample_id) != set(training.sample_id) or not frame.equals(training.reset_index(drop=True)):
        raise ValueError('El CSV de ajuste no coincide con los pacientes autorizados del fold')
    if not frame.split.eq('train').all() or frame.fold.eq(cfg['CV_FOLD']).any():
        raise ValueError('No se permite ajustar estadísticas con validación/test')
    hist = np.zeros((3,511),dtype=np.int64)
    for index,row in enumerate(frame.itertuples(),1):
        arrays=[]
        for column in ['path_pre','path_early','path_late']:
            with Image.open(project_path(cfg['DATA_DIR'])/getattr(row,column)) as image:
                if image.mode != 'L' or image.size != (cfg['IMAGE_SIZE'],cfg['IMAGE_SIZE']):
                    raise ValueError('Se requieren PNG 8 bit gris con el tamaño configurado')
                arrays.append(np.asarray(image,dtype=np.int16))
        pre,early,late=arrays
        channels=[pre,early-pre,late-pre] if cfg['INPUT_MODE']=='enhancement' else arrays
        for c,values in enumerate(channels):
            hist[c]+=np.bincount((values.ravel()+255).astype(np.int64),minlength=511)
        if index%500==0:
            print(f'Estadísticas fold {cfg["CV_FOLD"]}: {index}/{len(frame)} cortes train',flush=True)
    low=[_quantile(h,.01) for h in hist]
    high=[_quantile(h,.99) for h in hist]
    values=np.arange(-255,256,dtype=np.float64)/255
    means=[];stds=[]
    for c,h in enumerate(hist):
        clipped=np.clip(values,low[c],high[c]) if cfg['USE_CLIPPING'] else values
        mean=float(np.dot(h,clipped)/h.sum())
        std=float(np.sqrt(np.dot(h,(clipped-mean)**2)/h.sum()))
        if not np.isfinite(std) or std<=0:
            raise ValueError('Canal con desviación nula/no finita en train')
        means.append(mean);stds.append(std)
    cfg.update(CHANNEL_P01=low,CHANNEL_P99=high)
    signature={k:cfg[k] for k in ['INPUT_MODE','USE_CLIPPING','CHANNEL_P01','CHANNEL_P99']}
    result=dict(means=means,stds=stds,preprocessing=signature,n_samples=len(frame),
                n_patients=frame.patient_id.nunique(),train_csv_sha256=canonical_csv_hash(cfg['TRAIN_CSV']),
                diagnostic_partial=False,scope='smoke_only' if cfg['SMOKE_TEST'] else 'official_fold_train_only',
                fold=cfg['CV_FOLD'],method='png8_histogram_511_linear_quantiles_v1')
    with Path(output).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    return result


def artifact_hash(path):
    path=Path(path)
    if path.suffix=='.csv':
        return canonical_csv_hash(path)
    if path.suffix=='.json':
        content=json.dumps(json.loads(path.read_text(encoding='utf-8')),sort_keys=True).encode()
    else:
        content=path.read_bytes().replace(b'\r\n',b'\n')
    return hashlib.sha256(content).hexdigest()
