"""Prueba técnica acotada: batch fijo, gradientes y estimación de coste CPU."""
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.configuration import resolve_config, project_path
from src.dataset import BreastDCEDataset
from src.model import BreastCancerCNN
from src.experiment import (set_seed, select_device, create_optimizer, create_criterion,
                            create_run_directory, environment_info)


def main():
    cfg = resolve_config(dict(RUN_NAME="technical_fixed_batch", NUM_WORKERS=0, BATCH_SIZE=4))
    set_seed(cfg["SEED"])
    device = select_device(cfg)
    dataset = BreastDCEDataset(cfg["TRAIN_CSV"], cfg["DATA_DIR"], config=cfg)
    original_count = len(dataset)
    chosen = []
    for label in [0, 1]:
        candidates = dataset.df[dataset.df.pCR == label].sort_values(["patient_id", "sample_id"])
        chosen.extend(candidates.drop_duplicates("patient_id").head(2).sample_id.tolist())
    dataset.df = dataset.df[dataset.df.sample_id.isin(chosen)].reset_index(drop=True)
    start = time.perf_counter()
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    load_seconds = time.perf_counter() - start
    images, labels = batch["image"].to(device), batch["label"].to(device)
    model = BreastCancerCNN(cfg).to(device)
    criterion = create_criterion(cfg, device)
    optimizer = create_optimizer(model, cfg)
    directory = create_run_directory(cfg)
    dataset.df.to_csv(directory / "fixed_batch.csv", index=False)
    (directory / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (directory / "environment.json").write_text(json.dumps(environment_info(device), indent=2), encoding="utf-8")
    # Comparar siempre en modo eval para aislar el efecto de la actualización.
    model.eval()
    with torch.no_grad():
        initial = criterion(model(images).squeeze(1), labels).item()
    history, timings = [], []
    changed = False
    started = time.perf_counter()
    for step in range(1, 41):
        step_start = time.perf_counter()
        model.train()
        before = next(model.parameters()).detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        if not torch.isfinite(images).all() or not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Valores no finitos")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Gradientes ausentes o no finitos")
        grad_norm = sum(float(g.square().sum()) for g in gradients) ** .5
        if grad_norm == 0:
            raise RuntimeError("Gradiente nulo")
        optimizer.step()
        changed = changed or not torch.equal(before, next(model.parameters()).detach())
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Pesos no finitos")
        timings.append(time.perf_counter() - step_start)
        model.eval()
        with torch.no_grad():
            evaluation_logits = model(images).squeeze(1)
            fixed_loss = criterion(evaluation_logits, labels).item()
            probabilities = torch.sigmoid(evaluation_logits).cpu().tolist()
        history.append(dict(step=step, train_loss=loss.item(), fixed_eval_loss=fixed_loss,
                            grad_norm=grad_norm, step_seconds=timings[-1]))
        if step >= 10 and fixed_loss < initial * .80:
            break
        if time.perf_counter() - started > 25:
            break
    median_step = float(np.median(timings[1:] or timings))
    # Aproximación lineal: comunicar su incertidumbre, no usarla como benchmark de GPU.
    full_train = median_step * original_count / 4
    full_validation = median_step * 2200 / 4 * .5
    summary = dict(status="diagnostico_tecnico_parcial", initial_fixed_loss=initial,
                   final_fixed_loss=fixed_loss, minimum_fixed_loss=min(x["fixed_eval_loss"] for x in history),
                   steps=len(history), parameters_changed=changed, gradients_finite=True,
                   loss_reduced=fixed_loss < initial * .95,
                   load_batch_seconds=load_seconds, median_train_step_seconds=median_step,
                   estimated_full_epoch_seconds=full_train + full_validation,
                   estimated_four_two_epoch_runs_seconds=8 * (full_train + full_validation),
                   estimate_note="Extrapolación CPU batch 4; batch 16, E/S y caché pueden cambiar el tiempo",
                   probabilities=probabilities, labels=labels.cpu().tolist(), sample_ids=batch["sample_id"],
                   device=str(device), max_rss_note="Ver diagnóstico de memoria del informe; no se reservó el dataset completo")
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0] + list(range(1, len(history) + 1)), [initial] + [x["fixed_eval_loss"] for x in history])
    ax.set(xlabel="Actualización del batch fijo", ylabel="Loss en modo eval", title="Diagnóstico de capacidad de aprendizaje")
    fig.savefig(directory / "fixed_batch_loss.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(run=str(directory), **summary), indent=2), flush=True)
    if not summary["loss_reduced"] or not changed:
        raise RuntimeError("La prueba de aprendizaje falló; detener comparativas")


if __name__ == "__main__":
    main()
