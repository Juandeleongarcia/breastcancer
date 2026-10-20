"""Ajuste streaming de estadísticas: solo train, después de clipping y antes de z-score."""
import argparse
import json
import time

import numpy as np
from torch.utils.data import DataLoader

from src.configuration import resolve_config, preset_config, load_config, project_path, file_hash
from src.dataset import BreastDCEDataset


def calculate_statistics(config=None, max_samples=None):
    cfg = resolve_config(config)
    dataset = BreastDCEDataset(cfg["TRAIN_CSV"], cfg["DATA_DIR"], config=cfg,
                               standardize=False, augment=False)
    if "split" not in dataset.df or not dataset.df.split.eq("train").all():
        raise ValueError("Las estadísticas solo pueden ajustarse con train")
    # Exigir pertenencia a la partición de entrenamiento interna, nunca validación.
    import pandas as pd
    authorized = pd.read_csv(project_path("metadata/internal_train.csv"))
    if not set(dataset.df.sample_id).issubset(set(authorized.sample_id)):
        raise ValueError("El CSV contiene muestras ajenas al train interno")
    total = len(dataset)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples debe ser positivo")
        dataset.df = dataset.df.iloc[:max_samples].reset_index(drop=True)
    sums, squares, minima, maxima = np.zeros(3), np.zeros(3), np.full(3, np.inf), np.full(3, -np.inf)
    counts = np.zeros(3, dtype=np.int64)
    low_counts, high_counts = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    start = time.perf_counter()
    last_report = start
    processed = 0
    # Estadísticas con numpy float64; no se acumulan todas las imágenes en memoria.
    loader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], num_workers=cfg["NUM_WORKERS"], shuffle=False)
    for batch in loader:
        if cfg.get("DEADLINE_UNIX") and time.time() >= cfg["DEADLINE_UNIX"]:
            raise TimeoutError("Presupuesto agotado durante el ajuste de estadísticas; no se exporta un ajuste incompleto")
        array = batch["image"].numpy().astype(np.float64)
        processed += len(array)
        sums += array.sum(axis=(0, 2, 3))
        squares += np.square(array).sum(axis=(0, 2, 3))
        counts += array.shape[0] * array.shape[2] * array.shape[3]
        minima = np.minimum(minima, array.min(axis=(0, 2, 3)))
        maxima = np.maximum(maxima, array.max(axis=(0, 2, 3)))
        for c in range(3):
            low_counts[c] += np.isclose(array[:, c], cfg["CHANNEL_P01"][c], atol=1e-7).sum()
            high_counts[c] += np.isclose(array[:, c], cfg["CHANNEL_P99"][c], atol=1e-7).sum()
        if time.perf_counter() - last_report >= 30:
            print(f"Estadísticas: {processed}/{len(dataset)} cortes de train", flush=True)
            last_report = time.perf_counter()
    means = sums / counts
    stds = np.sqrt(np.maximum(squares / counts - means ** 2, 0))
    if not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Estadísticas no finitas o desviación nula")
    return dict(means=means.tolist(), stds=stds.tolist(), minima=minima.tolist(), maxima=maxima.tolist(),
                fraction_at_clip_low=(low_counts / counts).tolist(),
                fraction_at_clip_high=(high_counts / counts).tolist(),
                clipping_note="Fracciones en los límites, no estimación exacta de información perdida",
                preprocessing={k: cfg[k] for k in ["INPUT_MODE", "USE_CLIPPING", "CHANNEL_P01", "CHANNEL_P99"]},
                n_samples=len(dataset), diagnostic_partial=len(dataset) < total,
                train_csv_sha256=file_hash(cfg["TRAIN_CSV"]), seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list("ABCD"), default="A")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cfg = preset_config(args.preset)
    if args.config:
        cfg.update(load_config(args.config))
    cfg["NUM_WORKERS"] = args.workers
    import torch
    torch.set_num_threads(cfg["TORCH_NUM_THREADS"])
    result = calculate_statistics(cfg, args.max_samples)
    if args.output:
        path = project_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
