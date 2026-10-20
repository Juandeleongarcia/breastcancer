"""Reanudación en límites de época; serialización restringida y ramas inmutables."""
import hashlib
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from src.configuration import file_hash, project_path

COMPATIBILITY_KEYS = [
    "INPUT_CHANNELS", "CONV_CHANNELS", "CONVS_PER_BLOCK", "KERNEL_SIZE", "PADDING", "POOL_SIZE",
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
    result["csv_sha256"] = {"train": file_hash(cfg["TRAIN_CSV"]), "validation": file_hash(cfg["VAL_CSV"])}
    sources = ["src/model.py", "src/dataset.py", "src/experiment.py", "src/checkpointing.py", "src/metrics.py", "src/configuration.py", "src/augmentation.py"]
    result["source_sha256"] = {name: hashlib.sha256(project_path(name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for name in sources}
    return result


def experiment_key(cfg):
    payload = dict(contract(cfg), requested_epochs=cfg["EPOCHS"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    saved = checkpoint["training_contract"]
    differences = [key for key in expected if expected[key] != saved.get(key)]
    if differences:
        raise ValueError("Reanudación incompatible: " + ", ".join(differences))
    if cfg["EPOCHS"] <= checkpoint["epoch"]:
        raise ValueError("EPOCHS es el objetivo total y debe superar la última época completada")
    if cfg["USE_EARLY_STOPPING"] and checkpoint["epochs_without_improvement"] >= cfg["PATIENCE"]:
        raise ValueError("El checkpoint ya alcanzó early stopping; no se reinicia su contador")
    return checkpoint
