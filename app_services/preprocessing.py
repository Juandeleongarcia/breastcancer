"""Reutilizar los operadores reales del Dataset, sin CSV ficticios ni etiquetas."""
import numpy as np
import torch
from types import SimpleNamespace

from src.configuration import resolve_config
from src.dataset import BreastDCEDataset


def temporal_channels(phases):
    arrays = torch.from_numpy(np.asarray(phases).copy()).float()/255.
    return BreastDCEDataset._build_channels(SimpleNamespace(config={'INPUT_MODE':'enhancement'}),*arrays)


def validate_preprocessing(raw):
    required = ["INPUT_CHANNELS", "IMAGE_SIZE", "INPUT_MODE", "USE_CLIPPING",
                "USE_STANDARDIZATION", "CHANNEL_P01", "CHANNEL_P99", "CHANNEL_MEANS", "CHANNEL_STDS",
                "STATS_PROVENANCE", "STATS_PREPROCESSING"]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError("Falta el contrato de preprocesamiento: " + ", ".join(missing))
    if raw["INPUT_CHANNELS"] != 3 or raw["IMAGE_SIZE"] != 256 or raw["INPUT_MODE"] not in ["enhancement", "raw"]:
        raise ValueError("El modelo requiere tres canales, 256 × 256 y un modo DCE conocido")
    if any(not isinstance(raw[key],bool) for key in ['USE_CLIPPING','USE_STANDARDIZATION']):
        raise ValueError('Clipping y estandarización deben ser booleanos')
    for name in ["CHANNEL_P01", "CHANNEL_P99", "CHANNEL_MEANS", "CHANNEL_STDS"]:
        values = np.asarray(raw[name], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Estadísticas de tres canales no finitas o incompletas: " + name)
    if raw["USE_CLIPPING"] and not (np.asarray(raw["CHANNEL_P01"]) < np.asarray(raw["CHANNEL_P99"])).all():
        raise ValueError("Los límites de clipping no son válidos")
    if raw["USE_STANDARDIZATION"]:
        if raw["STATS_PROVENANCE"] != "verified_train_fit" or min(raw["CHANNEL_STDS"]) <= 0:
            raise ValueError("Se requieren estadísticas verificadas de train con desviaciones positivas")
        signature = {k: raw[k] for k in ["INPUT_MODE", "USE_CLIPPING", "CHANNEL_P01", "CHANNEL_P99"]}
        if signature != raw["STATS_PREPROCESSING"]:
            raise ValueError("Las estadísticas no corresponden a esta representación y clipping")
    return resolve_config(raw)


class TrainingTransform:
    # Métodos protegidos leídos, nunca modificados. Las pruebas contrastan el
    # resultado bit a bit con BreastDCEDataset.__getitem__ y sus PNG reales.
    _build_channels = BreastDCEDataset._build_channels
    _clip_channels = BreastDCEDataset._clip_channels
    _standardize = BreastDCEDataset._standardize

    def __init__(self, config):
        self.config = validate_preprocessing(config)

    def __call__(self, phases):
        array = np.asarray(phases)
        if array.shape != (3,256,256) or array.dtype != np.uint8:
            raise ValueError("El corte debe contener PRE/EARLY/LATE uint8 de 256 × 256")
        pre, early, late = torch.from_numpy(array.copy()).float()/255.
        image = self._build_channels(pre, early, late)
        return self._standardize(self._clip_channels(image))
