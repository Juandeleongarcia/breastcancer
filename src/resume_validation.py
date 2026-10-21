"""Preflight de versión para las interfaces públicas de continuación."""
import json
from pathlib import Path
import re

import torch
from src.run_layout import run_root, technical_file


def validate_resume_environment(checkpoint):
    checkpoint = Path(checkpoint)
    environment = technical_file(run_root(checkpoint), "environment.json")
    if not environment.exists():
        raise ValueError("Para reanudar, conservar también internals/environment.json (o details/environment.json histórico) junto al run de origen")
    saved = json.loads(environment.read_text(encoding="utf-8"))["packages"]["torch"]
    family = lambda version: tuple(re.match(r"(\d+)\.(\d+)", str(version)).groups())
    if family(saved) != family(torch.__version__):
        raise ValueError(f"PyTorch incompatible para continuar: checkpoint {saved}, entorno {torch.__version__}. "
                         "Iniciar un experimento nuevo con el entorno GPU existente; no sustituir PyTorch automáticamente")
