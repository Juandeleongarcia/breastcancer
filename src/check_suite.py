"""Preflight sin entrenamiento: particiones, estadísticas, archivos y CNN sintéticas."""
import argparse
import json
from pathlib import PurePosixPath

import pandas as pd
import torch

from src.batch_train import check_device, resolve_job
from src.configuration import load_config, project_path, file_hash
from src.model import BreastCancerCNN
from src.experiment import select_device, set_seed


def require_rocm():
    if not getattr(torch.version, 'hip', None) or not torch.cuda.is_available():
        raise RuntimeError('Se requiere PyTorch ROCm con GPU AMD accesible; no se permite fallback CPU/CUDA NVIDIA')


def verify_suite(path='config/phase6/rocm20_suite.json', device='cpu', synthetic=True, only=None):
    suite = load_config(path)
    if suite.get('phase') == 8:
        from src.phase8 import verify_phase8_suite
        return verify_phase8_suite(path, device, synthetic, only)
    if only:
        raise ValueError('--only en preflight está disponible para Fase 8')
    entries = suite['experiments']
    configs = {e['id']: resolve_job(e['config'], device) for e in entries}
    expected = {'BASE_SPATIAL': set(), 'WEIGHT_DECAY': {'WEIGHT_DECAY'},
                'REGIONAL_2X2': {'SPATIAL_GRID'}, 'GENTLE_AUG': {'USE_AUGMENTATION', 'AUGMENTATION_MODE'},
                'DROPOUT_050': {'DROPOUT'}}
    phase7 = suite.get('phase') == 7
    if phase7:
        expected = {'SPATIAL_CONTROL': set(),
                    'SPATIAL_GENTLE_AUG': {'USE_AUGMENTATION', 'AUGMENTATION_MODE'},
                    'SPATIAL_SMALL': {'CONV_CHANNELS'},
                    'SPATIAL_SCHEDULER': {'SCHEDULER_PATIENCE'}}
    if set(configs) != set(expected):
        raise ValueError('La batería debe contener exactamente los modelos acordados para esta fase')
    base = configs['SPATIAL_CONTROL' if phase7 else 'BASE_SPATIAL']
    if phase7 and any(base[k] != v for k, v in dict(CONV_CHANNELS=[32,64,128],
            CONVS_PER_BLOCK=2, SPATIAL_GRID=1, DROPOUT=.4, OPTIMIZER='adamw',
            LEARNING_RATE=.0005, WEIGHT_DECAY=.0005, USE_BATCH_NORM=True,
            USE_AUGMENTATION=False, SEED=42, BATCH_SIZE=16,
            USE_WEIGHTED_LOSS=False, SCHEDULER='plateau', SCHEDULER_PATIENCE=2).items()):
        raise ValueError('Control Fase 7 distinto del protocolo acordado')
    if phase7 and (configs['SPATIAL_SMALL']['CONV_CHANNELS'] != [16,32,64]
            or configs['SPATIAL_SCHEDULER']['SCHEDULER_PATIENCE'] != 3
            or configs['SPATIAL_GENTLE_AUG']['AUGMENTATION_MODE'] != 'gentle_affine'):
        raise ValueError('Variante Fase 7 distinta del contraste acordado')
    counts = verify_partitions(base)
    checks = []
    for name, cfg in configs.items():
        changed = {k for k in base if k not in {'RUN_NAME', 'COMPARISON_PARENT'} and cfg[k] != base[k]}
        if changed != expected[name]:
            raise ValueError(f'Diferencias no autorizadas en {name}: {changed}')
        if (cfg['EPOCHS'] != 20 or cfg['PATIENCE'] != (8 if phase7 else 5) or not cfg['USE_EARLY_STOPPING']
                or cfg['PRIMARY_METRIC'] != 'patient_auc' or cfg['STATS_PROVENANCE'] != 'verified_train_fit'
                or cfg['SMOKE_TEST'] or cfg['MAX_TRAIN_BATCHES'] or cfg['MAX_VAL_BATCHES']):
            raise ValueError('Protocolo incompatible: '+name)
        if phase7 and (not cfg['EVALUATE_TRAIN'] or cfg.get('TRAIN_PATIENTS_PER_CLASS')):
            raise ValueError('Fase 7 requiere evaluación completa de train: '+name)
        parameters = check_network(cfg) if synthetic else None
        checks.append(dict(id=name, changes=sorted(changed), parameters=parameters, passed=True))
    return dict(partitions=counts, models=checks, synthetic_only=True,
                hip_build=getattr(torch.version, 'hip', None), clinical_training_performed=False)


def verify_partitions(base):
    """Solo train/validación interna: entidades, rutas y correspondencia de fases."""
    train = pd.read_csv(project_path(base['TRAIN_CSV']))
    val = pd.read_csv(project_path(base['VAL_CSV']))
    if set(train.patient_id) & set(val.patient_id):
        raise ValueError('Pacientes compartidos entre train y validación')
    paths = lambda frame: set(frame[['path_pre', 'path_early', 'path_late']].to_numpy().ravel())
    if paths(train) & paths(val):
        raise ValueError('Imágenes compartidas entre train y validación')
    counts = {}
    for name, frame, patients, slices, csv in [
            ('train', train, 877, 8745, base['TRAIN_CSV']),
            ('validation', val, 220, 2200, base['VAL_CSV'])]:
        if (frame.patient_id.nunique() != patients or len(frame) != slices or frame.sample_id.duplicated().any()
                or not frame.split.eq('train').all() or not frame.pCR.isin([0, 1]).all()
                or not frame.groupby('patient_id').pCR.nunique().eq(1).all()):
            raise ValueError('Partición interna inválida: '+name)
        patient_labels = frame[['patient_id','pCR']].drop_duplicates().pCR
        counts[name] = dict(patients=patients, slices=slices, sha256=file_hash(csv),
                            patient_classes={str(k):int(v) for k,v in patient_labels.value_counts().items()},
                            slice_classes={str(k):int(v) for k,v in frame.pCR.value_counts().items()},
                            majority_accuracy=float(patient_labels.value_counts(normalize=True).max()))
        for column in ['path_pre', 'path_early', 'path_late']:
            for relative in frame[column].unique():
                relative = str(relative)
                portable = PurePosixPath(relative)
                if '\\' in relative or portable.is_absolute() or '..' in portable.parts:
                    raise ValueError('Ruta no portable o fuera del dataset')
                if not (project_path(base['DATA_DIR'])/relative).is_file():
                    raise FileNotFoundError('Falta una imagen de train/validación: '+relative)
        for row in frame.itertuples():
            phases = [PurePosixPath(getattr(row, k)) for k in ['path_pre','path_early','path_late']]
            if (any(p.parent.name != str(row.patient_id) for p in phases)
                    or any(not p.name.endswith('_'+phase+'.png') for p,phase in zip(phases,['PRE','EARLY','LATE']))
                    or len({p.stem.rsplit('_',1)[0] for p in phases}) != 1):
                raise ValueError('Las fases no corresponden al mismo paciente/corte')
    return counts


def check_network(cfg):
    """Solo forward/backward sintético; no optimizer.step ni datos clínicos."""
    set_seed(cfg['SEED'])
    actual = select_device(cfg)
    model = BreastCancerCNN(cfg).to(actual)
    images = torch.randn(2, 3, cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], device=actual)
    logits = model(images).squeeze(1)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0.,1.], device=actual)).backward()
    if logits.shape != (2,) or not torch.isfinite(logits).all() or not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Forward/backward no finito')
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='config/phase6/rocm20_suite.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--require-rocm', action='store_true')
    parser.add_argument('--only', nargs='+', help='Preflight de trabajos concretos de Fase 8')
    args = parser.parse_args()
    if args.require_rocm:
        if args.device != 'cuda':
            parser.error('--require-rocm requiere --device cuda (también en AMD)')
        require_rocm()
    check_device(args.device)
    result = verify_suite(args.suite, args.device, only=args.only)
    print(json.dumps(result, indent=2))
    if result.get('errors'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
