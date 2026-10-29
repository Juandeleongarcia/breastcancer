"""Configuración de la base visual: sin modelos, pesos o factories de inferencia."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE = 'Prototipo académico de investigación. No destinado a diagnóstico ni decisiones clínicas.'
PENDING_ITEMS = ('Arquitectura definitiva','Número de parámetros','Checkpoint final',
                 'Configuración de inferencia','Métricas definitivas','Resultados del test')


@dataclass(frozen=True)
class Workspace:
    results_campaign: str = ''


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def load_workspace(filename=None):
    candidate = filename or os.environ.get('BREASTDCEDL_APP_CONFIG') or ROOT/'models/app_config.json'
    path = local_path(candidate)
    if not path.exists():
        if filename or os.environ.get('BREASTDCEDL_APP_CONFIG'):
            raise ValueError('No se encuentra la configuración visual de la aplicación')
        return Workspace()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('results',{}),dict):
        raise ValueError('La configuración visual debe contener un objeto results')
    campaign = data.get('results',{}).get('campaign','')
    if not isinstance(campaign,str):raise ValueError('La ruta de resultados debe ser texto')
    # Los antiguos campos models/inference no seleccionan modelos ni activan pesos.
    return Workspace(results_campaign=campaign)
