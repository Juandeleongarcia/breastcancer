"""Reanudación en límites de época; serialización restringida y ramas inmutables."""
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from src.configuration import file_hash, project_path, canonical_csv_hash, csv_hash_variants

COMPATIBILITY_KEYS = [
    "SPATIAL_GRID", "INPUT_CHANNELS", "CONV_CHANNELS", "CONVS_PER_BLOCK", "KERNEL_SIZE", "PADDING", "POOL_SIZE",
    "USE_BATCH_NORM", "DROPOUT", "IMAGE_SIZE", "INPUT_MODE", "USE_CLIPPING", "CHANNEL_P01", "CHANNEL_P99",
    "USE_STANDARDIZATION", "CHANNEL_MEANS", "CHANNEL_STDS", "USE_AUGMENTATION", "BATCH_SIZE", "SEED",
    "OPTIMIZER", "LEARNING_RATE", "WEIGHT_DECAY", "USE_WEIGHTED_LOSS", "CLASSIFICATION_THRESHOLD",
    "SCHEDULER", "SCHEDULER_FACTOR", "SCHEDULER_PATIENCE", "SCHEDULER_MIN_LR", "PRIMARY_METRIC",
    "USE_EARLY_STOPPING", "PATIENCE", "SMOKE_TEST", "SMOKE_PATIENTS_PER_CLASS", "MAX_TRAIN_BATCHES", "MAX_VAL_BATCHES",
    "NUM_WORKERS", "TORCH_NUM_THREADS",
    "AUGMENTATION_MODE", "AUGMENT_PROBABILITY", "AUGMENT_DEGREES", "AUGMENT_TRANSLATE", "AUGMENT_SCALE_DELTA",
]


def contract(cfg):
    result = {key: cfg[key] for key in COMPATIBILITY_KEYS}
    result["TRAIN_PATIENTS_PER_CLASS"] = cfg.get("TRAIN_PATIENTS_PER_CLASS")
    result["csv_sha256"] = {"train": canonical_csv_hash(cfg["TRAIN_CSV"]), "validation": canonical_csv_hash(cfg["VAL_CSV"])}
    sources = ["src/model.py", "src/dataset.py", "src/experiment.py", "src/checkpointing.py", "src/metrics.py", "src/configuration.py", "src/augmentation.py", "src/run_layout.py"]
    result["source_sha256"] = {name: hashlib.sha256(project_path(name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for name in sources}
    return result


def experiment_key(cfg):
    payload = dict(contract(cfg), requested_epochs=cfg["EPOCHS"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def compatible_experiment_keys(cfg):
    """Mantener omisión de históricos solo para las fuentes auditadas y grid 1."""
    keys = {experiment_key(cfg)}
    migration_path = project_path("src/layout_compatibility.json")
    if migration_path.is_file():
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        current = contract(cfg)
        if current["source_sha256"] == migration["simple_layout_sources"]:
            prior = [dict(sources=sources, has_spatial_grid=False) for sources in
                     [migration["previous_sources"], *migration.get("compatible_source_sets", [])]]
            for entry in prior + migration.get("compatible_contracts", []):
                if not entry["has_spatial_grid"] and cfg["SPATIAL_GRID"] != 1:
                    continue
                legacy = dict(current, source_sha256=entry["sources"])
                if not entry["has_spatial_grid"]:
                    legacy.pop("SPATIAL_GRID")
                for train_sha, val_sha in product(csv_hash_variants(cfg["TRAIN_CSV"]), csv_hash_variants(cfg["VAL_CSV"])):
                    payload = dict(legacy, requested_epochs=cfg["EPOCHS"],
                                   csv_sha256={"train": train_sha, "validation": val_sha})
                    keys.add(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return keys


def capture_rng(generator):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(),
                 int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if device.type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Número de GPU distinto al checkpoint; no se restaura RNG de forma silenciosa")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume(path, cfg):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version", 0) < 3 or not checkpoint.get("resume_supported"):
        raise ValueError("Checkpoint sin estado completo reanudable. Usar un checkpoint Fase 4 de límite de época")
    if checkpoint.get("epoch_incomplete"):
        raise ValueError("Los pesos de media época son diagnósticos; reanudar desde recovery_epoch_start")
    expected = contract(cfg)
    saved = dict(checkpoint["training_contract"])
    saved.setdefault("SPATIAL_GRID", 1)  # Ausencia histórica = pooling 1x1.
    differences = [key for key in expected if expected[key] != saved.get(key)]
    if "csv_sha256" in differences:
        hashes = saved.get("csv_sha256", {})
        if (set(hashes) == {"train", "validation"}
                and hashes["train"] in csv_hash_variants(cfg["TRAIN_CSV"])
                and hashes["validation"] in csv_hash_variants(cfg["VAL_CSV"])):
            differences.remove("csv_sha256")
    if "source_sha256" in differences:
        # Migración acotada: solo la pareja EXACTA de fuentes auditadas de este cambio.
        # CSV, arquitectura, configuración, optimizador y scheduler siguen verificándose.
        migration_path = project_path("src/layout_compatibility.json")
        if migration_path.is_file():
            migration = json.loads(migration_path.read_text(encoding="utf-8"))
            if (saved.get("source_sha256") in [migration["previous_sources"], *migration.get("compatible_source_sets", []), *[entry["sources"] for entry in migration.get("compatible_contracts", [])]]
                    and expected["source_sha256"] == migration["simple_layout_sources"]):
                differences.remove("source_sha256")
    if differences:
        raise ValueError("Reanudación incompatible: " + ", ".join(differences))
    if cfg["EPOCHS"] <= checkpoint["epoch"]:
        raise ValueError("EPOCHS es el objetivo total y debe superar la última época completada")
    if cfg["USE_EARLY_STOPPING"] and checkpoint["epochs_without_improvement"] >= cfg["PATIENCE"]:
        raise ValueError("El checkpoint ya alcanzó early stopping; no se reinicia su contador")
    return checkpoint
