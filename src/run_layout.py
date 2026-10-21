"""Lectura de runs simples y de formatos históricos, sin migrar resultados."""
from pathlib import Path


def run_root(checkpoint):
    path = Path(checkpoint)
    return path.parent.parent if path.parent.name in {"checkpoints", "internals"} else path.parent


def technical_file(directory, name):
    directory = Path(directory)
    candidates = [directory/"internals"/name, directory/"details"/name, directory/name]
    return next((p for p in candidates if p.is_file()), candidates[0])


def checkpoint_directories(directory):
    directory = Path(directory)
    return [directory, directory/"checkpoints", directory/"internals"]


def summary_paths(root):
    root = Path(root)
    return sorted(set(root.glob("RUN_*/internals/summary.json")) | set(root.glob("*/summary.json")))
