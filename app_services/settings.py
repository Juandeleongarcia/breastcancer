"""Rutas explícitas y locales; sin descubrir ni seleccionar modelos automáticamente."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from app_services.architectures import ARCHITECTURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = tuple(ARCHITECTURES)
NOTICE = "Prototipo académico de investigación. No destinado a diagnóstico ni decisiones clínicas."


@dataclass(frozen=True)
class ModelPaths:
    checkpoint: str = ""
    campaign: str = ""
    config: str = ""


@dataclass(frozen=True)
class InferencePlan:
    """Plan futuro independiente de las arquitecturas que se comparan en la UI."""
    enabled: bool = False
    architecture: str = ""
    checkpoint: str = ""
    config: str = ""


def local_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def _settings_data(filename=None):
    candidate = filename or os.environ.get("BREASTDCEDL_APP_CONFIG") or ROOT/"models/app_config.json"
    path = local_path(candidate)
    if not path.exists():
        if filename or os.environ.get("BREASTDCEDL_APP_CONFIG"):
            raise FileNotFoundError("No existe la configuración de la app: " + str(path))
        return {'models':{}}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict):
        raise ValueError('La configuración de la app debe ser un objeto JSON')
    return data


def load_inference_plan(filename=None):
    values = _settings_data(filename).get('inference',{})
    fields = {'enabled','architecture','checkpoint','config'}
    if not isinstance(values,dict) or set(values)-fields:
        raise ValueError('Configuración de inferencia futura no válida')
    plan = InferencePlan(**values)
    if not isinstance(plan.enabled,bool) or any(not isinstance(getattr(plan,name),str) for name in fields-{'enabled'}):
        raise ValueError('Tipos inválidos en el plan de inferencia')
    if plan.enabled and (plan.architecture not in MODEL_IDS or not plan.checkpoint):
        raise ValueError('La activación futura requiere arquitectura y checkpoint explícitos')
    return plan


def load_settings(filename=None):
    result = {name: ModelPaths() for name in MODEL_IDS}
    data = _settings_data(filename)
    models = data.get('models',{})
    if not isinstance(models,dict) or set(models)-set(MODEL_IDS):
        raise ValueError("Configurar en models únicamente familias del catálogo experimental; otras campañas se abren por su ruta")
    for name, values in models.items():
        if not isinstance(values, dict) or set(values)-{"checkpoint", "campaign", "config"}:
            raise ValueError("Campos de rutas desconocidos en " + name)
        if any(not isinstance(value, str) for value in values.values()):
            raise ValueError("Las rutas deben ser cadenas de texto")
        result[name] = ModelPaths(**values)
    return result
