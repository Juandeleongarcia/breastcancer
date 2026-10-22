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
    info = json.loads(environment.read_text(encoding="utf-8"))
    saved = info["packages"]["torch"]
    family = lambda version: tuple(re.match(r"(\d+)\.(\d+)", str(version)).groups())
    if family(saved) != family(torch.__version__):
        raise ValueError(f"PyTorch incompatible para continuar: checkpoint {saved}, entorno {torch.__version__}. "
                         "Iniciar un experimento nuevo con el entorno GPU existente; no sustituir PyTorch automáticamente")

    saved_backend = info.get("backend")
    if saved_backend is None and info.get("device"):
        saved_backend = ("cpu" if info["device"] == "cpu" else
                         "cuda" if info.get("cuda_build") else "rocm")
    current_backend = "rocm" if getattr(torch.version, "hip", None) else "cuda" if torch.version.cuda else "cpu"
    if saved_backend not in [None, "cpu", current_backend]:
        raise ValueError(f"Backend incompatible para reanudar: {saved_backend} -> {current_backend}. Iniciar desde cero; no convertir estados RNG entre CUDA y ROCm")

    if info.get("hip_build") and info["hip_build"] != getattr(torch.version, "hip", None):
        raise ValueError("ROCm/HIP distinto al checkpoint: conservar el entorno original o iniciar desde cero")
