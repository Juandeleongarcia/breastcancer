"""Ejecutor local secuencial: omisión, errores, presupuesto y reanudación explícita."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import torch
import pandas as pd

from src.configuration import ROOT, load_config, resolve_config, project_path, apply_statistics
from src.checkpointing import experiment_key, load_resume, compatible_experiment_keys
from src.experiment import train, select_device
from src.compare_runs import comparison
from src.run_layout import summary_paths, technical_file, checkpoint_directories
from src.resume_validation import validate_resume_environment


def check_device(requested):
    cfg = resolve_config({"DEVICE": requested})
    device = select_device(cfg)
    model = torch.nn.Conv2d(3, 4, 3).to(device)
    x = torch.randn(2, 3, 16, 16, device=device)
    y = model(x)
    y.square().mean().backward()
    if not torch.isfinite(y).all() or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("La prueba de dispositivo produjo valores no finitos")
    if device.type == "cuda":
        torch.cuda.synchronize()
    info = dict(python=sys.executable, torch=str(torch.__version__), cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), actual_device=str(device),
                hip_build=getattr(torch.version, "hip", None),
                backend=("rocm" if getattr(torch.version, "hip", None) else "cuda" if torch.version.cuda else "cpu"),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
                forward_backward_passed=True)
    print(json.dumps(info, indent=2), flush=True)
    return info


def resolve_job(path, device, epochs=None):
    cfg = resolve_config(load_config(path))
    cfg["DEVICE"] = device
    if epochs is not None:
        cfg["EPOCHS"] = epochs
    if cfg.get("STATS_PATH"):
        cfg = apply_statistics(cfg, cfg["STATS_PATH"])
    return cfg


def completed_run(cfg):
    keys = compatible_experiment_keys(cfg)
    for path in reversed(summary_paths(project_path(cfg["RESULTS_DIR"]))):
        summary = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent.parent if path.parent.name == "internals" else path.parent
        if (summary.get("experiment_key") in keys and summary.get("status") == "entrenamiento_completo"
                and not summary.get("diagnostic_partial") and any((folder/"best_epoch.pt").exists() for folder in checkpoint_directories(directory))):
            return directory
    return None


def interrupted_checkpoint(cfg):
    candidates = []
    key = experiment_key(cfg)
    for directory in project_path(cfg["RESULTS_DIR"]).glob("RUN_*"):
        details = technical_file(directory, "config.json")
        if not details.exists():
            continue
        saved = resolve_config(load_config(details))
        if experiment_key(saved) != key:
            continue
        paths = {path for folder in checkpoint_directories(directory)
                 for pattern in ["epoch_*.pt", "recovery_epoch_start*.pt"] for path in folder.glob(pattern)}
        for path in paths:
            try:
                validate_resume_environment(path)
                payload = load_resume(path, cfg)
                candidates.append((payload["epoch"], path.stat().st_mtime_ns, path))
            except (ValueError, RuntimeError, OSError):
                continue
    return max(candidates)[2] if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite")
    parser.add_argument("--only", nargs="+", help="Identificadores del suite, sin cambiar su configuración")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--check-device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--force-new", action="store_true", help="Repetir desde cero incluso si ya está completo")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.force_new and args.resume_interrupted:
        parser.error('--force-new y --resume-interrupted son acciones distintas; elegir una')
    if args.check_device:
        check_device(args.check_device)
        return
    if not args.suite:
        parser.error("Indicar --suite o --check-device")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes debe ser positivo")
    suite = load_config(args.suite)
    if suite.get("required_backend") == "rocm" and (not getattr(torch.version, "hip", None) or not torch.cuda.is_available() or args.device == "cpu"):
        raise RuntimeError("Esta batería exige GPU AMD con PyTorch ROCm; no se permite entrenamiento CPU/CUDA NVIDIA")
    entries = suite["experiments"]
    if args.only:
        if set(args.only)-{x["id"] for x in entries}:
            parser.error("Hay identificadores --only que no existen en el suite")
        entries = [entry for entry in entries if entry["id"] in args.only]
    check_device(args.device)  # CUDA solicitada nunca cae silenciosamente a CPU.
    directory = project_path("models") / f"BATCH_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    started = time.time()
    deadline = started+args.max_minutes*60 if args.max_minutes else None
    results = []
    result_roots = {"models"}
    def save_results():
        (directory/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        frame = pd.concat([comparison(root) for root in result_roots], ignore_index=True)
        rows = []
        for entry in entries:
            item = next((r for r in reversed(results) if r["id"] == entry["id"]), {})
            matching = frame[frame.Run.eq(Path(item["run"]).name)] if item.get("run") else pd.DataFrame()
            row = matching.iloc[0].to_dict() if not matching.empty else {}
            row.update(ID_suite=entry["id"], Estado_ejecucion=item.get("status", "pendiente"))
            if not matching.empty:
                row["Estado_modelo"] = row.get("Estado")
            row["Estado"] = item.get("status", "pendiente")
            rows.append(row)
        pd.DataFrame(rows).to_csv(directory/"comparison.csv", index=False)
    save_results()
    for entry in entries:
        if deadline and time.time() >= deadline-30:
            results.append(dict(id=entry["id"], status="pendiente_por_presupuesto"))
            continue
        try:
            cfg = resolve_job(entry["config"], args.device, args.epochs)
            if suite.get("phase") == 8:
                from src.phase8 import audit_job
                audited = audit_job(entry, cfg)
                print("Contraste verificado: " + json.dumps(audited), flush=True)
            result_roots.add(cfg["RESULTS_DIR"])
            done = None if args.force_new else completed_run(cfg)
            if done:
                results.append(dict(id=entry["id"], status="omitido_completo", run=str(done)))
                print(f"Omitido: {entry['id']} ya completo en {done}", flush=True)
                continue
            estimate = entry.get("estimated_cpu_seconds") if args.device == "cpu" else entry.get("estimated_gpu_seconds")
            if deadline and estimate and deadline-time.time() < estimate*1.10+30:
                results.append(dict(id=entry["id"], status="pendiente_por_estimacion", estimated_seconds=estimate))
                continue
            checkpoint = interrupted_checkpoint(cfg) if args.resume_interrupted and not args.force_new else None
            print(f"{entry['id']}: " + (f"reanudar desde {checkpoint}" if checkpoint else "experimento nuevo desde cero"), flush=True)
            cfg["DEADLINE_UNIX"] = deadline-30 if deadline else None
            run, summary = train(cfg, resume_from=checkpoint)
            results.append(dict(id=entry["id"], status=summary["status"], run=str(run),
                                resumed_from=str(checkpoint) if checkpoint else None))
            if summary["status"].startswith("interrumpido"):
                break
        except Exception as error:
            results.append(dict(id=entry["id"], status="error", error=str(error)))
            with (directory/"errors.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(dict(id=entry["id"], error=str(error), traceback=traceback.format_exc()))+"\n")
            print(f"ERROR {entry['id']}: {error}; registrado, continúo con el siguiente.", flush=True)
        finally:
            save_results()
    save_results()  # También si todos quedan pendientes antes de entrar en try/finally.
    (directory/"execution.json").write_text(json.dumps(dict(seconds=time.time()-started, results=results), indent=2), encoding="utf-8")
    print(f"Lote finalizado: {directory}", flush=True)


if __name__ == "__main__":
    main()
