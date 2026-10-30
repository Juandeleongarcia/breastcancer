"""Runtime web: artefactos verificados, recursos compartidos y entradas privadas sin caché."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch

from app_services.workspace import ROOT
from app_services.studies import study_from_files, PATTERN, PHASES
from app_services.preprocessing import TrainingTransform
from src.model import BreastCancerCNN
from src.phase13_model import SpatialSmallMultimodal
from src.phase13_clinical import ClinicalTransform
from src.phase13_inference import ImageTransform, predict_patient
from src.phase13_ensemble import average_patients

LABELS={'image':'SPATIAL_SMALL · imágenes','multimodal':'SPATIAL_SMALL multimodal · imágenes + clínica'}


def safe_file(root,name):
    if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name or ':' in name or name in ['.','..']:
        raise ValueError('Referencia de artefacto no válida')
    path=Path(root)/name
    if path.is_symlink():raise ValueError('No se admiten enlaces en los artefactos')
    return path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_package(content,expected,folder):
    if len(content)>64*1024*1024 or hashlib.sha256(content).hexdigest()!=expected:
        raise ValueError('El paquete no coincide con el checksum publicado')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    with ZipFile(BytesIO(content)) as archive:
        entries=archive.infolist()
        if len(entries)>40 or sum(item.file_size for item in entries)>128*1024*1024:
            raise ValueError('El paquete supera los límites autorizados')
        names=set()
        allowed=r'(catalog\.json|ATTRIBUTION\.txt|(image|multimodal)_fold_[0-4](\.pt|_history\.csv)|public_example_(PRE|EARLY|LATE)\.png)'
        for entry in entries:
            if (not re.fullmatch(allowed,entry.filename) or entry.filename in names
                    or stat.S_ISLNK(entry.external_attr>>16) or entry.flag_bits&1):
                raise ValueError('Contenido de paquete no permitido')
            names.add(entry.filename)
            safe_file(folder,entry.filename).write_bytes(archive.read(entry))
    return folder


def package_root():
    override=os.environ.get('BREASTDCEDL_BUNDLE_DIR')
    config_file=ROOT/'config/app_release.json'
    config=json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    candidate=Path(override) if override else ROOT/config.get('local_directory','models/app_bundle')
    if (candidate/'catalog.json').is_file():return candidate.resolve()
    if override:raise ValueError('La carpeta de artefactos configurada no contiene catalog.json')
    url=config.get('url','');expected=config.get('sha256','')
    if not url or not re.fullmatch('[a-f0-9]{64}',expected):
        raise ValueError('No hay un paquete de modelos configurado. Consultar STREAMLIT_GUIA.md')
    if not url.startswith('https://github.com/') or '/releases/download/' not in url:
        raise ValueError('Se requiere una URL HTTPS de release y checksum fijo, configurados por el administrador')
    cache=ROOT/'models/.app_cache';cache.mkdir(parents=True,exist_ok=True)
    target=cache/expected
    if (target/'catalog.json').is_file():return target
    response=requests.get(url,timeout=(10,60),stream=True)
    response.raise_for_status();data=bytearray()
    for chunk in response.iter_content(1024*1024):
        data.extend(chunk)
        if len(data)>64*1024*1024:raise ValueError('Descarga de artefactos demasiado grande')
    # Carpeta exclusiva por intento; no se escribe dentro de campañas históricas.
    from uuid import uuid4
    temporary=cache/(expected+'_'+uuid4().hex[:8])
    extract_package(bytes(data),expected,temporary)
    try:temporary.rename(target)
    except OSError:
        if (target/'catalog.json').is_file():return target
        raise
    return target


def read_catalog(root=None):
    root=Path(root) if root else package_root()
    data=json.loads(safe_file(root,'catalog.json').read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or set(data.get('modalities',{}))!={'image','multimodal'}:
        raise ValueError('Catálogo de modelos incompatible')
    for name,digest in data['files'].items():
        if sha(safe_file(root,name))!=digest:raise ValueError('Un artefacto no coincide con su checksum')
    for kind,entry in data['modalities'].items():
        if len(entry['members'])!=5 or [m['fold'] for m in entry['members']]!=list(range(5)):
            raise ValueError('Se requieren cinco checkpoints por modalidad')
        if any(m['sha256']!=data['files'].get(m['file']) for m in entry['members']):
            raise ValueError('Identidad de los checkpoints no verificable')
    return root,data


@dataclass
class Resource:
    members: list
    checksum: str
    kind: str
    device: torch.device
    lock: object


def load_resource(root,kind,device='cpu'):
    root,catalog=read_catalog(root)
    if kind not in LABELS:raise ValueError('Modalidad no disponible')
    if device not in ['cpu','cuda']:raise ValueError('Dispositivo no admitido')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('La GPU solicitada no está disponible')
    target=torch.device(device);torch.set_num_threads(4);members=[]
    for item in catalog['modalities'][kind]['members']:
        payload=torch.load(safe_file(root,item['file']),map_location='cpu',weights_only=True)
        if payload.get('schema')!='breastdcedl-inference-v1' or payload.get('kind')!=kind or payload['epoch']!=item['epoch']:
            raise ValueError('Checkpoint de inferencia incompatible')
        cfg=payload['config']
        if cfg['CLASSIFICATION_THRESHOLD']!=.5:raise ValueError('El umbral publicado debe permanecer fijo en 0,50')
        if kind=='multimodal':
            clinical=ClinicalTransform(payload['clinical_transform'])
            model=SpatialSmallMultimodal(cfg,clinical.dimension)
            processor=ImageTransform(cfg)
        else:clinical=None;model=BreastCancerCNN(cfg);processor=TrainingTransform(cfg)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        if sum(p.numel() for p in model.parameters())!=item['parameters']:raise ValueError('Número de parámetros incoherente')
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Pesos no finitos')
        model.to(target).eval()
        members.append(dict(model=model,config=cfg,clinical=clinical,images=processor,device=target,
            contract={'threshold':.5},metadata=item))
    return Resource(members,catalog['modalities'][kind]['checksum'],kind,target,threading.RLock())


def predict(study,resource,clinical_values=None):
    if study.synthetic:raise ValueError('Las formas sintéticas son solo para el explorador; no predicen pCR')
    if resource.kind=='multimodal' and (not isinstance(clinical_values,dict) or set(clinical_values)!={'age','HR_HER2_STATUS'}):
        raise ValueError('Indicar edad y subtipo, o declarar explícitamente su ausencia')
    began=time.perf_counter();frames={};samples=pd.DataFrame(dict(patient_id=['input']*len(study.slice_indices),
        sample_id=[f'input_z{i}' for i in range(len(study.slice_indices))]))
    per_slice=[]
    with resource.lock:
        for fold,bundle in enumerate(resource.members):
            bundle['model'].eval()
            if resource.kind=='multimodal':
                result=predict_patient(bundle,study.phases,clinical_values,'input')
                values=result['slices'].probability.tolist()
            else:
                values=[];size=bundle['config']['BATCH_SIZE']
                with torch.inference_mode():
                    for offset in range(0,len(study.slice_indices),size):
                        images=torch.stack([bundle['images'](item) for item in study.phases[offset:offset+size]])
                        logits=bundle['model'](images.to(resource.device)).squeeze(1)
                        if not torch.isfinite(logits).all():raise ValueError('La inferencia produjo logits no finitos')
                        values.extend(torch.sigmoid(logits).cpu().tolist())
            frame=pd.DataFrame(dict(patient_id='input',sample_id=samples.sample_id,probability=values))
            patient=frame.groupby('patient_id').agg(probability=('probability','mean'),n_slices=('sample_id','size')).reset_index()
            frames[fold]=patient;per_slice.append(values)
    mean=average_patients(frames,samples).iloc[0]
    if resource.device.type=='cuda':torch.cuda.synchronize(resource.device)
    elapsed=time.perf_counter()-began
    probability=float(mean.probability)
    return dict(patient_probability=probability,no_pcr_probability=1-probability,prediction=int(probability>=.5),
        threshold=.5,checksum=resource.checksum,kind=resource.kind,model=LABELS[resource.kind],
        n_models=5,n_slices=len(study.slice_indices),device=str(resource.device),seconds=elapsed,
        member_probabilities=[float(frames[f].probability.iloc[0]) for f in range(5)],
        slices=pd.DataFrame(dict(slice_index=study.slice_indices,probability=np.mean(per_slice,axis=0))),
        study_fingerprint=study.fingerprint)


def triplet(files):
    if set(files)!=set(PHASES) or any(value is None for value in files.values()):
        raise ValueError('Cargar PRE, EARLY y LATE antes de analizar')
    values=[];seen=set();ids=[]
    for phase in PHASES:
        name,content=files[phase]
        if not isinstance(content,bytes):raise ValueError('El contenido debe ser un archivo PNG, no una ruta')
        if Path(name).name!=name or '/' in name or '\\' in name or ':' in name:raise ValueError('No se admiten rutas del sistema')
        if name.casefold() in seen:raise ValueError('Hay un archivo duplicado en fases diferentes')
        seen.add(name.casefold())
        match=PATTERN.fullmatch(name)
        if match:
            if match['phase'].upper()!=phase:raise ValueError('La fase indicada por el nombre no corresponde al control de carga')
            ids.append((match['patient'],int(match['index'])))
        elif re.search(r'_(PRE|EARLY|LATE)\.png$',name,re.I) and not name.upper().endswith('_'+phase+'.PNG'):
            raise ValueError('La fase del archivo no corresponde a la seleccionada')
        values.append((f'input_z000_{phase}.png',content))
    if ids and len(set(ids))!=1:raise ValueError('Las fases pertenecen a pacientes o cortes distintos')
    return study_from_files(values)


def public_example(root=None):
    root,catalog=read_catalog(root)
    files={phase:(name,safe_file(root,name).read_bytes()) for phase,name in zip(PHASES,catalog['example']['files'])}
    return triplet(files)


def diagnostics():
    root,catalog=read_catalog()
    study=public_example(root)
    checks={}
    for kind in LABELS:
        models=load_resource(root,kind)
        checks[kind]=dict(checkpoints=len(models.members),checksum=models.checksum,device=str(models.device),
            parameters=[sum(p.numel() for p in m['model'].parameters()) for m in models.members])
    return dict(ready=True,modalities=checks,public_example_valid=study.phases.shape==(1,3,256,256),
                private_inputs_stored=False,external_test_reevaluated=False)


if __name__=='__main__':print(json.dumps(diagnostics(),indent=2))
