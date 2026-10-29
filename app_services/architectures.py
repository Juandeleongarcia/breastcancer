"""Perfiles calculados sobre las clases reales, sin modificar sus fuentes."""
from copy import deepcopy
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from src.configuration import load_config, resolve_config
from src.model import BreastCancerCNN
from src.phase11_model import DCEFourBlockCNN, BLOCK_CHANNELS, POOL_AFTER_BLOCK


def _small_contract(raw):
    expected = dict(CONV_CHANNELS=[16,32,64],CONVS_PER_BLOCK=2,SPATIAL_GRID=1,
                    USE_BATCH_NORM=True,KERNEL_SIZE=3,PADDING=1,POOL_SIZE=2)
    return (raw.get('ARCHITECTURE') in [None,'','SPATIAL_SMALL']
            and all(raw.get(key)==value for key,value in expected.items()))


def _four_contract(raw):
    return (raw.get('ARCHITECTURE')=='F11_CNN4' and raw.get('BLOCK_CHANNELS')==BLOCK_CHANNELS
            and raw.get('POOL_AFTER_BLOCK')==POOL_AFTER_BLOCK
            and raw.get('CONV_BIAS') is False and raw.get('DROPOUT')==.30)


@dataclass(frozen=True)
class ArchitectureAdapter:
    factory: Callable
    config_file: str
    accepts: Callable


# Catálogo experimental, no ranking ni selección del modelo definitivo.
ARCHITECTURES = {
    'SPATIAL_SMALL':ArchitectureAdapter(BreastCancerCNN,'config/phase9/SPATIAL_SMALL.json',_small_contract),
    'F11_CNN4':ArchitectureAdapter(DCEFourBlockCNN,'config/phase11/F11_CNN4.json',_four_contract)
}


def identify_architecture(raw):
    if isinstance(raw,dict):
        for name,adapter in ARCHITECTURES.items():
            if adapter.accepts(raw):return name
    raise ValueError('Arquitectura o contrato no admitidos por los adaptadores registrados')


def create_model(raw):
    kind = identify_architecture(raw)
    with torch.random.fork_rng(devices=[]):
        model = ARCHITECTURES[kind].factory(raw)
    return kind, model


@lru_cache(maxsize=2)
def _default_profile(kind):
    name = ARCHITECTURES[kind].config_file
    cfg = resolve_config(load_config(name))
    _, model = create_model(cfg)
    stages = [dict(name="Entrada DCE", shape="3 × 256 × 256", parameters=0,
                   description="PRE, EARLY−PRE y LATE−PRE; normalización ajustada en train.")]
    side = 256
    if hasattr(model,'blocks'):
        groups = list(model.blocks)
    else:
        groups = []
        current = []
        for layer in model.features:
            current.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                groups.append(nn.Sequential(*current));current = []
    for i, group in enumerate(groups, 1):
        conv = [m for m in group if isinstance(m, nn.Conv2d)]
        pool = any(isinstance(m, nn.MaxPool2d) for m in group)
        if pool: side //= 2
        stages.append(dict(name=f"Bloque {i}",shape=f"{conv[-1].out_channels} × {side} × {side}",
                           parameters=sum(v.numel() for v in group.parameters()),
                           description=f"Filtros: {' → '.join(str(m.out_channels) for m in conv)}. "
                           + "Conv 3×3 + BatchNorm + ReLU. " + ("MaxPool 2×2." if pool else "Sin pooling final.")))
    channels = cfg['CONV_CHANNELS'][-1]
    stages.extend([dict(name="Global Average Pool",shape=f"{channels} × 1 × 1",parameters=0,
                        description="Promedia cada mapa espacial sin introducir parámetros."),
                   dict(name="Dropout",shape=str(channels),parameters=0,
                        description=f"p={cfg['DROPOUT']:.2f}; regularización activa solo al entrenar."),
                   dict(name="Logit",shape="1",parameters=channels+1,
                        description="Linear → un logit. Sigmoid se aplica después para estimar pCR.")])
    return dict(id=kind,parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                stages=stages,config=cfg)


def architecture_profile(kind):
    if kind not in ARCHITECTURES:
        raise ValueError("Arquitectura desconocida")
    return deepcopy(_default_profile(kind))
