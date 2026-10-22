"""Ejecución incremental de la CNN existente y registro de ambos niveles."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.configuration import (ROOT, resolve_config, project_path, file_hash, load_config,
                               preset_config, apply_statistics)
from src.dataset import BreastDCEDataset
from src.model import BreastCancerCNN
from src.run_layout import run_root, checkpoint_directories
from src.checkpointing import (contract, experiment_key, capture_rng, restore_rng, atomic_save, load_resume)
from src.metrics import evaluate_predictions, save_evaluation_artifacts, plot_training_history, analyze_probabilities


class TrainingBudgetExceeded(RuntimeError):
    """Salida controlada: conservar el estado sin evaluar una validación incompleta."""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(cfg):
    device = cfg["DEVICE"]
    if device == "auto":
        device = "cuda" if cfg["USE_GPU"] and torch.cuda.is_available() else "cpu"
    if device not in ["cpu", "cuda"]:
        raise ValueError("DEVICE debe ser auto, cpu o cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada pero no disponible")
    torch.set_num_threads(cfg["TORCH_NUM_THREADS"])
    return torch.device(device)


def create_run_directory(config=None):
    cfg = resolve_config(config)
    root = project_path(cfg["RESULTS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    suffix = cfg.get("RUN_NAME") or "train"
    suffix = "".join(c if c.isalnum() or c in "_-" else "_" for c in suffix)
    stamp = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%Y%m%d_%H%M%S")
    directory = root / f"RUN_{stamp}_{suffix}_{uuid4().hex[:12]}"
    directory.mkdir(exist_ok=False)
    (directory / "internals").mkdir()
    return directory


def environment_info(device):
    versions = {}
    for package in ["torch", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib", "requests"]:
        versions[package] = importlib.metadata.version(package)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                device=str(device), cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                hip_build=getattr(torch.version, "hip", None),
                backend=("rocm" if getattr(torch.version, "hip", None) else "cuda" if torch.version.cuda else "cpu"),
                gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                git_commit=git.stdout.strip(), git_dirty=bool(status.stdout.strip()))


def calculate_pos_weight(csv_path):
    labels = pd.read_csv(project_path(csv_path)).pCR
    if not labels.isin([0, 1]).all() or not (labels == 1).any():
        raise ValueError("Se requieren etiquetas binarias y muestras pCR positivas")
    return float((labels == 0).sum() / (labels == 1).sum())


def create_optimizer(model, config=None):
    cfg = resolve_config(config)
    name = cfg["OPTIMIZER"].lower()
    choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    if name not in choices:
        raise ValueError(f"Optimizador desconocido: {name}")
    kwargs = dict(lr=cfg["LEARNING_RATE"], weight_decay=cfg["WEIGHT_DECAY"])
    if name == "sgd":
        kwargs["momentum"] = 0.9
    return choices[name](model.parameters(), **kwargs)


def create_criterion(cfg, device, pos_weight=None):
    if cfg["USE_WEIGHTED_LOSS"]:
        weight = calculate_pos_weight(cfg["TRAIN_CSV"]) if pos_weight is None else pos_weight
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    return nn.BCEWithLogitsLoss()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None, threshold=0.5, deadline=None):
    training = optimizer is not None
    model.train(training)
    rows, total_loss, count = [], 0.0, 0
    last_progress = time.perf_counter()
    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if deadline is not None and time.time() >= deadline:
                raise TrainingBudgetExceeded("Presupuesto agotado antes del siguiente batch")
            if max_batches is not None and index >= max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if not torch.isfinite(images).all():
                raise FloatingPointError("Entradas no finitas")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Logits/loss no finitos")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Gradientes no finitos")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Parámetros no finitos")
            probabilities = torch.sigmoid(logits.detach()).cpu().tolist()
            truth = labels.detach().cpu().tolist()
            rows.extend(dict(sample_id=str(sid), patient_id=str(pid), label=int(y), probability=float(p))
                        for sid, pid, y, p in zip(batch["sample_id"], batch["patient_id"], truth, probabilities))
            total_loss += loss.item() * len(labels)
            count += len(labels)
            if deadline is not None and time.time() >= deadline:
                raise TrainingBudgetExceeded("Presupuesto agotado después del batch")
            if time.perf_counter() - last_progress >= 30:
                print(f"{'Train' if training else 'Validación'}: {count}/{len(loader.dataset)} cortes; loss={total_loss/count:.5f}", flush=True)
                last_progress = time.perf_counter()
    if not count:
        raise ValueError("El DataLoader no produjo muestras")
    result = evaluate_predictions(rows, threshold)
    result.update(loss=total_loss / count, diagnostic_partial=count != len(loader.dataset))
    # Alias históricos: siempre métricas POR CORTE, etiquetadas también en slice.
    result.update({k: result["slice"][k] for k in ["accuracy", "auc", "sensitivity", "specificity"]})
    result.update(labels=[x["label"] for x in rows], probabilities=[x["probability"] for x in rows])
    return result


def validate(model, loader, criterion, device, config=None):
    cfg = resolve_config(config)
    result = run_epoch(model, loader, criterion, device, max_batches=cfg["MAX_VAL_BATCHES"],
                       threshold=cfg["CLASSIFICATION_THRESHOLD"], deadline=cfg.get("DEADLINE_UNIX"))
    result["diagnostic_partial"] = bool(result["diagnostic_partial"] or cfg["SMOKE_TEST"])
    return result


def select_patients(frame, n_per_class, seed):
    labels = frame[["patient_id", "pCR"]].drop_duplicates().sort_values("patient_id")
    ids = []
    for label in [0, 1]:
        group = labels[labels.pCR == label]
        if len(group) < n_per_class:
            raise ValueError("No hay pacientes suficientes de ambas clases")
        ids.extend(group.sample(n=n_per_class, random_state=seed).patient_id.tolist())
    return frame[frame.patient_id.isin(ids)].reset_index(drop=True)


def make_datasets(cfg):
    train = BreastDCEDataset(cfg["TRAIN_CSV"], cfg["DATA_DIR"], config=cfg, augment=True)
    val = BreastDCEDataset(cfg["VAL_CSV"], cfg["DATA_DIR"], config=cfg)
    if set(train.df.patient_id) & set(val.df.patient_id):
        raise ValueError("Fuga de pacientes entre train y validación")
    # Bloquear test en este flujo incluso si se cambia accidentalmente VAL_CSV.
    if "split" not in train.df or "split" not in val.df or not train.df.split.eq("train").all() or not val.df.split.eq("train").all():
        raise ValueError("Entrenamiento y validación interna solo pueden proceder del train público")
    if cfg.get("TRAIN_PATIENTS_PER_CLASS"):
        train.df = select_patients(train.df, cfg["TRAIN_PATIENTS_PER_CLASS"], cfg["SEED"])
    if cfg["SMOKE_TEST"]:
        train.df = select_patients(train.df, cfg["SMOKE_PATIENTS_PER_CLASS"], cfg["SEED"])
        val.df = select_patients(val.df, cfg["SMOKE_PATIENTS_PER_CLASS"], cfg["SEED"])
    for dataset, key in [(train, "EXPECTED_TRAIN_PATIENTS"), (val, "EXPECTED_VAL_PATIENTS")]:
        if cfg.get(key) is not None and dataset.df.patient_id.nunique() != cfg[key]:
            raise ValueError(f"{key} no coincide con la partición de pacientes")
    return train, val


def scalar_metrics(result):
    return {"loss": result["loss"], "diagnostic_partial": result["diagnostic_partial"],
            "slice": result["slice"], "patient": result["patient"]}


def train(config=None, resume_from=None):
    raw_resume = torch.load(project_path(resume_from), map_location="cpu", weights_only=True) if resume_from else None
    cfg = resolve_config(raw_resume["config"] if config is None and raw_resume else config)
    if cfg.get("STATS_PATH") and not resume_from:
        cfg = apply_statistics(cfg, cfg["STATS_PATH"])
    if cfg["USE_STANDARDIZATION"] and cfg.get("STATS_PROVENANCE") == "requires_fit":
        raise ValueError("Este preprocesamiento necesita estadísticas train: usar calcular_stats.py y --stats")
    signature = {k: cfg[k] for k in ["INPUT_MODE", "USE_CLIPPING", "CHANNEL_P01", "CHANNEL_P99"]}
    if cfg["USE_STANDARDIZATION"] and cfg["STATS_PREPROCESSING"] != signature:
        raise ValueError("La representación o el clipping no corresponde a las estadísticas; ajustar train con calcular_stats.py")
    set_seed(cfg["SEED"])
    device = select_device(cfg)
    resume_payload = load_resume(project_path(resume_from), cfg) if resume_from else None
    if resume_payload and resume_payload["device"] != str(device) and not cfg.get("ALLOW_DEVICE_CHANGE"):
        raise ValueError("Cambio de dispositivo al reanudar: usar --allow-device-change y aceptar una trayectoria numérica diferente")
    train_ds, val_ds = make_datasets(cfg)
    name_cfg = dict(cfg)
    if resume_from:
        name_cfg["RUN_NAME"] = f"{cfg.get('RUN_NAME') or 'train'}_RESUME_E{resume_payload['epoch']:03d}"
    directory = create_run_directory(name_cfg)
    details = directory / "internals"
    checkpoints = directory
    cfg["DIAGNOSTIC_PARTIAL"] = bool(cfg["SMOKE_TEST"] or cfg["MAX_TRAIN_BATCHES"] or cfg["MAX_VAL_BATCHES"] or cfg.get("TRAIN_PATIENTS_PER_CLASS"))
    (details / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (directory / "config.py").write_text("# Snapshot exacto; no modifica config/config.py\n# Train online: pesos cambiantes, dropout/augmentation según configuración.\n# Validación: pesos fijos al final de época, modo eval, sin augmentation.\n" +
        "\n".join(f"{key} = {value!r}" for key, value in cfg.items()), encoding="utf-8")
    env = environment_info(device)
    (details / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    # Un solo archivo comprimido conserva las fuentes sin un árbol técnico adicional.
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(details / "sources.zip", "w", compression=ZIP_DEFLATED) as sources:
        for source in ["src/model.py", "src/dataset.py", "src/experiment.py", "src/metrics.py",
                       "src/configuration.py", "src/checkpointing.py", "src/augmentation.py", "src/run_layout.py"]:
            sources.write(project_path(source), source)
    if cfg["DIAGNOSTIC_PARTIAL"]:
        for name, ds in [("train", train_ds), ("validation", val_ds)]:
            ds.df.to_csv(details / f"{name}_selection.csv", index=False)
    hashes = {name: file_hash(cfg[key]) for name, key in [("train", "TRAIN_CSV"), ("validation", "VAL_CSV")]}
    generator = torch.Generator().manual_seed(cfg["SEED"])
    loader_kw = dict(batch_size=cfg["BATCH_SIZE"], num_workers=cfg["NUM_WORKERS"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    model = BreastCancerCNN(cfg).to(device)
    weight = calculate_pos_weight(cfg["TRAIN_CSV"])
    criterion = create_criterion(cfg, device, weight)
    optimizer = create_optimizer(model, cfg)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["SCHEDULER_FACTOR"], patience=cfg["SCHEDULER_PATIENCE"],
        min_lr=cfg["SCHEDULER_MIN_LR"], threshold=0.0) if cfg["SCHEDULER"] == "plateau" else None)
    history, best, best_epoch, waiting = [], -math.inf, 0, 0
    best_result, best_weights = None, None
    start_epoch, prior_seconds = 1, 0.0
    last_tr, last_va = None, None
    if resume_payload:
        model.load_state_dict(resume_payload["model_state_dict"])
        if scheduler:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        history = resume_payload["history"]
        best, best_epoch = resume_payload["best_patient_auc"], resume_payload["best_epoch"]
        waiting = resume_payload["epochs_without_improvement"]
        start_epoch = resume_payload["epoch"] + 1
        prior_seconds = resume_payload["elapsed_training_seconds"]
        best_weights = resume_payload["best_model_state_dict"]
        last_tr, last_va = resume_payload["train_metrics"], resume_payload["val_metrics"]
        if best_weights is not None:
            best_result = evaluate_predictions(resume_payload["best_slice_predictions"], cfg["CLASSIFICATION_THRESHOLD"])
            best_result.update(loss=resume_payload["best_validation_loss"], diagnostic_partial=cfg["DIAGNOSTIC_PARTIAL"])
            # Conservar también el estado completo del mejor checkpoint heredado.
            candidates = [folder/name for folder in checkpoint_directories(run_root(project_path(resume_from)))
                          for name in [f"epoch_{best_epoch:03d}.pt", "best_epoch.pt"]]
            source_best = None
            for path in candidates:
                if path.is_file():
                    candidate = torch.load(path, map_location="cpu", weights_only=True)
                    weights = candidate.get("model_state_dict", {})
                    if (candidate.get("epoch") == best_epoch and set(weights) == set(best_weights)
                            and all(torch.equal(weights[key], best_weights[key]) for key in weights)):
                        source_best = path
                        break
            if source_best is not None:
                import shutil
                shutil.copy2(source_best, checkpoints / "best_epoch.pt")
            else:
                atomic_save(dict(config=cfg, epoch=best_epoch, model_state_dict=best_weights, pos_weight=weight,
                                 csv_sha256=hashes, resume_supported=False, inherited_best=True), checkpoints / "best_epoch.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        (details / "resume.json").write_text(json.dumps(dict(source=str(resume_from),
            completed_epoch=resume_payload["epoch"], source_device=resume_payload["device"],
            destination_device=str(device), exact_backend=resume_payload["device"] == str(device)), indent=2), encoding="utf-8")
        restore_rng(resume_payload["rng_state"], generator, device)
    budget_stopped = False
    stop_reason = None
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Run: {directory}\nDispositivo: {device}; diagnóstico parcial: {cfg['DIAGNOSTIC_PARTIAL']}", flush=True)
    def snapshot(epoch, complete=True):
        return dict(schema_version=3, resume_supported=complete, epoch_incomplete=not complete,
                    config=cfg, device=str(device), epoch=epoch, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                    pos_weight=weight, train_metrics=last_tr, val_metrics=last_va,
                    best_epoch=best_epoch, best_patient_auc=best, epochs_without_improvement=waiting,
                    training_contract=contract(cfg), history=list(history), csv_sha256=hashes,
                    rng_state=capture_rng(generator), elapsed_training_seconds=prior_seconds+time.perf_counter()-start,
                    best_model_state_dict=best_weights,
                    best_slice_predictions=best_result["slice_predictions"].to_dict("records") if best_result else [],
                    best_validation_loss=best_result["loss"] if best_result else None)

    for epoch in range(start_epoch, cfg["EPOCHS"] + 1):
        epoch_start = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        recovery = details / "recovery_epoch_start.pt"
        atomic_save(snapshot(epoch-1), recovery)
        try:
            tr = run_epoch(model, train_loader, criterion, device, optimizer, cfg["MAX_TRAIN_BATCHES"], cfg["CLASSIFICATION_THRESHOLD"], cfg.get("DEADLINE_UNIX"))
            va = validate(model, val_loader, criterion, device, cfg)
        except (TrainingBudgetExceeded, KeyboardInterrupt) as error:
            budget_stopped = True
            stop_reason = "interrumpido_por_usuario" if isinstance(error, KeyboardInterrupt) else "interrumpido_por_presupuesto"
            partial = snapshot(epoch, complete=False)
            partial["recovery_checkpoint"] = str(recovery)
            atomic_save(partial, details / f"interrupted_epoch_{epoch:03d}.pt")
            print("Interrupción: pesos parciales conservados; reanudar desde el checkpoint de inicio de época.", flush=True)
            break
        except Exception:
            partial = snapshot(epoch, complete=False)
            partial["recovery_checkpoint"] = str(recovery)
            atomic_save(partial, details / f"failed_epoch_{epoch:03d}.pt")
            raise
        row = dict(epoch=epoch, learning_rate=lr_used, train_loss=tr["loss"], val_loss=va["loss"],
                   epoch_time_seconds=time.perf_counter() - epoch_start,
                   diagnostic_partial=cfg["DIAGNOSTIC_PARTIAL"],
                   train_metric_protocol="online_changing_weights_dropout_augmentation",
                   validation_metric_protocol="end_of_epoch_eval_no_augmentation")
        for partition, result in [("train", tr), ("val", va)]:
            for level in ["slice", "patient"]:
                for key, value in result[level].items():
                    if key != "confusion_matrix":
                        row[f"{partition}_{level}_{key}"] = value
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        score = va["patient"]["auc"]
        improved = math.isfinite(score) and score > best
        if improved:
            best, best_epoch, waiting = score, epoch, 0
            best_result = va
            best_weights = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            waiting += 1
        if scheduler:
            scheduler.step(score)
        row["next_learning_rate"] = optimizer.param_groups[0]["lr"]
        # Guardar el LR posterior al scheduler en el mismo historial que se reanuda.
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        last_tr, last_va = scalar_metrics(tr), scalar_metrics(va)
        checkpoint = snapshot(epoch)
        checkpoint.update(val_auc=va["slice"]["auc"], val_patient_auc=score)
        # Cada época completa se conserva, independientemente del antiguo SAVE_EVERY.
        atomic_save(checkpoint, checkpoints / f"epoch_{epoch:03d}.pt")
        # El checkpoint de época ya contiene todo el estado; evitar duplicar last_epoch.
        recovery.unlink(missing_ok=True)
        if improved:
            atomic_save(checkpoint, checkpoints / "best_epoch.pt")
            save_evaluation_artifacts(best_result, directory, details, simple=True)
        plot_training_history(directory / "history.csv", directory)
        print(f"Época {epoch}: train loss={tr['loss']:.5f}; val loss={va['loss']:.5f}; "
              f"AUC paciente={score:.5f}; AUC corte={va['slice']['auc']:.5f}; {row['epoch_time_seconds']:.2f}s", flush=True)
        if cfg["USE_EARLY_STOPPING"] and waiting >= cfg["PATIENCE"]:
            break
    if not best_epoch:
        if budget_stopped:
            summary = dict(run=directory.name, experiment=cfg.get("RUN_NAME"), status=stop_reason,
                           epochs_completed=0, seconds=time.perf_counter()-start, diagnostic_partial=True)
            (details / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return directory, summary
        raise RuntimeError("No se pudo seleccionar un checkpoint con AUC de paciente finito")
    save_evaluation_artifacts(best_result, directory, details, simple=True)
    diagnostics = analyze_probabilities(best_result, details, details)
    plot_training_history(directory / "history.csv", directory)
    summary = dict(run=directory.name, experiment=cfg.get("RUN_NAME"), cnn=cfg["CONV_CHANNELS"],
                   weighted_loss=cfg["USE_WEIGHTED_LOSS"], clipping=cfg["USE_CLIPPING"],
                   epochs_completed=len(history), best_epoch=best_epoch,
                   diagnostic_partial=cfg["DIAGNOSTIC_PARTIAL"],
                   status=stop_reason if budget_stopped else ("preliminar_train_parcial" if cfg["DIAGNOSTIC_PARTIAL"] else "entrenamiento_completo"),
                   layout_version=5, requested_epochs=cfg["EPOCHS"], input_mode=cfg["INPUT_MODE"],
                   interpretation=cfg.get("INTERPRETATION", "experimento_exploratorio"),
                   convolutions_per_block=cfg["CONVS_PER_BLOCK"], scheduler=cfg["SCHEDULER"],
                   experiment_key=experiment_key(cfg), resumed_from=str(resume_from) if resume_from else None,
                   resume_epoch=start_epoch-1, epochs_executed_here=len(history)-(start_epoch-1),
                   elapsed_training_seconds_total=prior_seconds+time.perf_counter()-start,
                   stopped_early=bool(cfg["USE_EARLY_STOPPING"] and waiting>=cfg["PATIENCE"]),
                   learning_rate=cfg["LEARNING_RATE"], dropout=cfg["DROPOUT"], standardization=cfg["USE_STANDARDIZATION"],
                   validation_complete=not best_result["diagnostic_partial"], comparison_parent=cfg.get("COMPARISON_PARENT"),
                   validation=scalar_metrics(best_result), seconds=time.perf_counter() - start,
                   train_patients=train_ds.df.patient_id.nunique(), validation_patients=val_ds.df.patient_id.nunique(),
                   pos_weight=weight, csv_sha256=hashes,
                   train_metric_note="Métricas online: parámetros y dropout cambian durante la época",
                   cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    (details / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    patient = best_result["patient"]
    patient_probs = diagnostics["patient"]
    analysis = (f"# Análisis del run {directory.name}\n\n"
                f"Mejor época: {best_epoch}; AUC paciente {patient['auc']:.6f}; AUC corte {best_result['slice']['auc']:.6f}.\n\n"
                f"Recall {patient['recall']:.4f}; F1 {patient['f1']:.4f}; accuracy {patient['accuracy']:.4f}.\n"
                f"Fracción predicha pCR: {patient_probs['predicted_positive_fraction']:.4f}; errores: {patient_probs['n_errors']}.\n\n"
                f"Distribuciones: {json.dumps(patient_probs['by_label'], ensure_ascii=False)}\n\n"
                "Umbral fijo 0.50. El AUC usa puntuaciones continuas y no se modifica mediante el umbral.\n"
                "Las épocas iniciales diagnostican aprendizaje temprano; no establecen superioridad estable.\n")
    (details / "analysis.md").write_text(analysis, encoding="utf-8")
    print(f"Mejor época por AUC de paciente: {best_epoch}; resultados: {directory}", flush=True)
    return directory, summary


def main():
    parser = argparse.ArgumentParser(description="CNN propia pCR/no pCR; selección por paciente")
    parser.add_argument("--preset", choices=list("ABCD"))
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--stats")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", help="Continuación explícita, siempre en una carpeta nueva")
    parser.add_argument("--allow-device-change", action="store_true")
    args = parser.parse_args()
    if args.resume and args.preset:
        parser.error("--resume no se combina con --preset; no mezclar continuación y experimento nuevo")
    cfg = (resolve_config(torch.load(project_path(args.resume), map_location="cpu", weights_only=True)["config"])
           if args.resume else (preset_config(args.preset) if args.preset else resolve_config()))
    if args.resume:
        cfg["DEADLINE_UNIX"] = None
        cfg["ALLOW_DEVICE_CHANGE"] = args.allow_device_change
    if args.config:
        cfg.update(load_config(args.config))
    if args.smoke:
        cfg.update(SMOKE_TEST=True, EPOCHS=2, NUM_WORKERS=0)
    for arg, key in [("epochs", "EPOCHS"), ("workers", "NUM_WORKERS"), ("stats", "STATS_PATH"), ("device", "DEVICE")]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
