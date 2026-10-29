"""Inferencia CPU independiente: pesos propios, contrato completo y sin entrenamiento."""
from dataclasses import dataclass
import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import torch

from app_services.architectures import create_model
from app_services.preprocessing import TrainingTransform, validate_preprocessing
from app_services.settings import local_path
from src.configuration import load_config


@dataclass
class ModelBundle:
    architecture: str
    config: dict
    model: torch.nn.Module
    checkpoint_sha256: str
    epoch: int
    parameters: int


def load_checkpoint(filename, expected_architecture=None, config_file=None):
    path = local_path(filename)
    if not path.is_file() or path.stat().st_size > 128*1024*1024:
        raise ValueError("Checkpoint ausente o superior a 128 MiB")
    content = path.read_bytes()
    try:
        payload = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("No se pudo cargar de forma restringida este checkpoint propio") from error
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Se espera un diccionario con model_state_dict, config y epoch")
    raw = payload.get("config")
    if config_file:
        companion = load_config(local_path(config_file))
        if raw is not None and any(raw.get(k) != v for k,v in companion.items()):
            raise ValueError("La configuración externa contradice la guardada en el checkpoint")
        raw = raw or companion
    if not isinstance(raw, dict):
        raise ValueError("Falta configuración exacta; aportar el JSON original con estadísticas de train")
    architecture_fields = ['CONV_CHANNELS','CONVS_PER_BLOCK','SPATIAL_GRID','KERNEL_SIZE','PADDING',
                           'POOL_SIZE','USE_BATCH_NORM','DROPOUT']
    if any(key not in raw for key in architecture_fields):
        raise ValueError('Falta el contrato exacto de arquitectura; no se adivinan sus valores')
    cfg = validate_preprocessing(raw)
    if (payload.get("epoch_incomplete") or cfg.get("SMOKE_TEST") or cfg.get("MAX_TRAIN_BATCHES")
            or cfg.get("MAX_VAL_BATCHES") or cfg.get("TRAIN_PATIENTS_PER_CLASS")):
        raise ValueError("No se integra un checkpoint parcial o diagnóstico como modelo de demostración real")
    if not 0 < int(payload.get("epoch", 0)) or "CLASSIFICATION_THRESHOLD" not in raw:
        raise ValueError("Falta época válida o umbral documentado")
    if not np.isfinite(raw['CLASSIFICATION_THRESHOLD']) or not 0 <= raw['CLASSIFICATION_THRESHOLD'] <= 1:
        raise ValueError("Umbral de clasificación inválido")
    saved = payload.get("training_contract", {})
    keys = ["INPUT_CHANNELS","CONV_CHANNELS","CONVS_PER_BLOCK","SPATIAL_GRID","KERNEL_SIZE","PADDING",
            "POOL_SIZE","USE_BATCH_NORM","DROPOUT","IMAGE_SIZE","INPUT_MODE","USE_CLIPPING",
            "CHANNEL_P01","CHANNEL_P99","USE_STANDARDIZATION","CHANNEL_MEANS","CHANNEL_STDS",
            "CLASSIFICATION_THRESHOLD"]
    if saved and any(key in saved and saved[key] != cfg.get(key) for key in keys):
        raise ValueError("El contrato del checkpoint y su configuración no coinciden")
    kind, model = create_model(cfg)
    if expected_architecture and kind != expected_architecture:
        raise ValueError(f"El checkpoint contiene {kind}, no {expected_architecture}")
    if kind == 'F11_CNN4' and saved.get('architecture', {}).get('id', kind) != kind:
        raise ValueError("Identidad de arquitectura incoherente en el checkpoint")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("Los pesos no corresponden exactamente a esta arquitectura") from error
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("El checkpoint contiene pesos o buffers no finitos")
    model.cpu().eval()
    return ModelBundle(kind,cfg,model,hashlib.sha256(content).hexdigest(),int(payload['epoch']),
                       sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict_study(study, bundle, progress=None):
    if study.synthetic:
        raise ValueError("La demostración sintética es solo para el visor; no genera predicciones")
    processor = TrainingTransform(bundle.config)
    values = []
    batch_size = int(bundle.config['BATCH_SIZE'])
    if not 1 <= batch_size <= 256:
        raise ValueError("Batch size del checkpoint fuera de rango")
    bundle.model.eval()
    with torch.inference_mode():
        for start in range(0,len(study.slice_indices),batch_size):
            images = torch.stack([processor(item) for item in study.phases[start:start+batch_size]])
            logits = bundle.model(images).squeeze(1)
            if logits.ndim != 1 or not torch.isfinite(logits).all():
                raise ValueError("La inferencia no produjo logits finitos por corte")
            values.extend(torch.sigmoid(logits).cpu().tolist())
            if progress: progress(min(start+batch_size,len(study.slice_indices)),len(study.slice_indices))
    frame = pd.DataFrame(dict(patient_id=study.patient_id,slice_index=study.slice_indices,probability=values))
    # Misma operación que src.metrics.evaluate_predictions, sin inventar etiquetas.
    score = float(frame.groupby('patient_id',sort=True).probability.mean().iloc[0])
    threshold = float(bundle.config['CLASSIFICATION_THRESHOLD'])
    return dict(patient_probability=score,no_pcr_probability=1-score,prediction=int(score>=threshold),
                threshold=threshold,slices=frame,checkpoint_sha256=bundle.checkpoint_sha256,
                study_fingerprint=study.fingerprint,architecture=bundle.architecture)
