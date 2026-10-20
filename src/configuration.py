"""Configuraciones explícitas, portables y sin ejecutar código de un run."""
import ast
import hashlib
import json
from pathlib import Path

from config import config as defaults

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def file_hash(path):
    return hashlib.sha256(project_path(path).read_bytes()).hexdigest()


def resolve_config(config=None):
    values = {k: v for k, v in vars(defaults).items() if k.isupper()}
    if config is not None:
        values.update(config if isinstance(config, dict) else vars(config))
    if values.get("EARLY_STOPPING_PATIENCE") is not None:
        values["PATIENCE"] = values["EARLY_STOPPING_PATIENCE"]
    if values.get("RANDOM_SEED") is not None:
        values["SEED"] = values["RANDOM_SEED"]
    if values["PRIMARY_METRIC"] != "patient_auc":
        raise ValueError("La selección principal debe usar patient_auc")
    if values["INPUT_CHANNELS"] != 3 or not values["CONV_CHANNELS"]:
        raise ValueError("Se requieren tres fases y al menos un bloque")
    if values["CONVS_PER_BLOCK"] not in [1, 2]:
        raise ValueError("CONVS_PER_BLOCK debe ser 1 (baseline) o 2 (espacial)")
    if values["SCHEDULER"] not in ["none", "plateau"]:
        raise ValueError("SCHEDULER debe ser none o plateau")
    if values['AUGMENTATION_MODE'] not in ['legacy_flips', 'gentle_affine']:
        raise ValueError('AUGMENTATION_MODE no válido')
    if not 0 <= values['AUGMENT_PROBABILITY'] <= 1:
        raise ValueError('AUGMENT_PROBABILITY debe estar en [0,1]')
    for key, maximum in [('AUGMENT_DEGREES', 10), ('AUGMENT_TRANSLATE', .05), ('AUGMENT_SCALE_DELTA', .05)]:
        if not 0 <= values[key] <= maximum:
            raise ValueError(f'{key} fuera del rango geométrico suave permitido')
    for key in ["BATCH_SIZE", "EPOCHS", "SAVE_EVERY", "PATIENCE", "TORCH_NUM_THREADS"]:
        if values[key] < 1:
            raise ValueError(f"{key} debe ser positivo")
    for key in ["MAX_TRAIN_BATCHES", "MAX_VAL_BATCHES"]:
        if values[key] is not None and values[key] < 1:
            raise ValueError(f"{key} debe ser positivo o None")
    return values


def load_config(path):
    path = project_path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    values[target.id] = ast.literal_eval(node.value)
    return values


def preset_config(name="A", smoke=False):
    cfg = resolve_config()
    if name not in "ABCD" or len(name) != 1:
        raise ValueError("Experimento esperado: A, B, C o D")
    cfg.update(EPOCHS=2, RUN_NAME=name, SMOKE_TEST=smoke)
    if name == "B":
        cfg["CONV_CHANNELS"] = [32, 64]
    if name == "C":
        cfg.update(USE_CLIPPING=False, STATS_PATH=None, STATS_PROVENANCE="requires_fit")
    if name == "D":
        cfg["USE_WEIGHTED_LOSS"] = False
    return cfg


def apply_statistics(cfg, path):
    stats = load_config(path)
    if stats["diagnostic_partial"]:
        raise ValueError("No se admiten estadísticas parciales para normalizar runs")
    expected = {k: cfg[k] for k in ["INPUT_MODE", "USE_CLIPPING", "CHANNEL_P01", "CHANNEL_P99"]}
    if stats["preprocessing"] != expected or stats["train_csv_sha256"] != file_hash(cfg["TRAIN_CSV"]):
        raise ValueError("Las estadísticas no corresponden al train o preprocesamiento")
    cfg.update(CHANNEL_MEANS=stats["means"], CHANNEL_STDS=stats["stds"],
               STATS_PATH=str(path), STATS_PROVENANCE="verified_train_fit",
               STATS_PREPROCESSING=stats["preprocessing"])
    return cfg
