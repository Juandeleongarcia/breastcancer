"""Plan controlado de diez experimentos y presupuesto compartido de CPU."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
from torch.utils.data import DataLoader

from src.configuration import resolve_config, project_path
from src.dataset import BreastDCEDataset
from src.experiment import (train, select_device, set_seed, create_criterion,
                            create_optimizer, run_epoch)
from src.model import BreastCancerCNN


def plan():
    base = resolve_config(dict(EPOCHS=3, SMOKE_TEST=False, MAX_TRAIN_BATCHES=None,
                               MAX_VAL_BATCHES=None, NUM_WORKERS=0, DEVICE="cpu",
                               CLASSIFICATION_THRESHOLD=.5, TRAIN_PATIENTS_PER_CLASS=None))
    variants = [
        ("P3_01", "baseline original", None, {}),
        ("P3_02", "sin clipping adicional", "P3_01", {"USE_CLIPPING": False, "STATS_PROVENANCE": "requires_fit"}),
        ("P3_03", "learning rate 0.0001", "P3_01", {"LEARNING_RATE": .0001}),
        ("P3_04", "learning rate 0.001", "P3_01", {"LEARNING_RATE": .001}),
        ("P3_05", "dropout 0.20", "P3_01", {"DROPOUT": .20}),
        ("P3_06", "dropout 0.00", "P3_01", {"DROPOUT": .00}),
        ("P3_07", "pérdida no ponderada", "P3_01", {"USE_WEIGHTED_LOSS": False}),
        ("P3_08", "dos bloques", "P3_01", {"CONV_CHANNELS": [32, 64]}),
        ("P3_09", "sin z-score global", "P3_01", {"USE_STANDARDIZATION": False}),
        ("P3_10", "fases raw sin clipping", "P3_02", {"USE_CLIPPING": False, "INPUT_MODE": "raw", "STATS_PROVENANCE": "requires_fit"}),
    ]
    result = []
    for name, description, parent, changes in variants:
        cfg = dict(base, **changes, RUN_NAME=name, COMPARISON_PARENT=parent)
        result.append(dict(name=name, description=description, comparison_parent=parent,
                           config=cfg, status="pendiente_GPU"))
    return result


def benchmark(cfg, deadline=None):
    set_seed(cfg["SEED"])
    device = select_device(cfg)
    model = BreastCancerCNN(cfg).to(device)
    criterion = create_criterion(cfg, device)
    optimizer = create_optimizer(model, cfg)
    times = {}
    for label, csv, opt in [("train", cfg["TRAIN_CSV"], optimizer), ("validation", cfg["VAL_CSV"], None)]:
        dataset = BreastDCEDataset(csv, cfg["DATA_DIR"], config=cfg)
        loader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], num_workers=0, shuffle=False)
        started = time.perf_counter()
        run_epoch(model, loader, criterion, device, optimizer=opt, max_batches=8, deadline=deadline)
        elapsed = time.perf_counter()-started
        times[label] = dict(measured_seconds=elapsed, measured_samples=128,
                            estimated_epoch_seconds=elapsed * len(dataset) / 128)
    times["estimated_epoch_seconds"] = sum(times[k]["estimated_epoch_seconds"] for k in ["train", "validation"])
    times["estimated_3_epoch_run_seconds"] = times["estimated_epoch_seconds"]*3 + 15
    times["estimated_ten_runs_seconds"] = 10 * times["estimated_3_epoch_run_seconds"]
    times["note"] = "Extrapolación de ocho batches por fase, incluye E/S; incertidumbre por caché y carga CPU"
    return times


def prepare():
    directory = project_path("models") / f"PHASE3_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    directory.mkdir(exist_ok=False)
    start = time.time()
    state = dict(start_unix=start, deadline_unix=start+3600, budget_seconds=3600, completed=[])
    (directory / "budget.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    entries = plan()
    for entry in entries:
        (directory / f"{entry['name']}.json").write_text(json.dumps(entry["config"], indent=2), encoding="utf-8")
    (directory / "plan.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    measured = benchmark(entries[0]["config"])
    (directory / "benchmark.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
    print(json.dumps(dict(session=str(directory), benchmark=measured), indent=2), flush=True)


def execute(session, name, decision, stats=None):
    directory = project_path(session)
    state = json.loads((directory / "budget.json").read_text(encoding="utf-8"))
    if any(item["experiment"] == name for item in state["completed"]):
        raise ValueError("Este experimento ya se ejecutó en la sesión; no se repite automáticamente")
    cfg = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if cfg["EPOCHS"] not in [3, 4] or cfg["SMOKE_TEST"] or cfg["MAX_VAL_BATCHES"] is not None:
        raise ValueError("La campaña requiere 3–4 épocas y validación completa")
    remaining = state["deadline_unix"]-time.time()
    measured = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    estimate = measured["estimated_3_epoch_run_seconds"]
    if state["completed"]:
        estimate = max(item["seconds"] for item in state["completed"])
    specific_benchmark = directory / f"benchmark_{name}.json"
    if specific_benchmark.exists():
        estimate = json.loads(specific_benchmark.read_text(encoding="utf-8"))["estimated_3_epoch_run_seconds"]
    # No asumir ahorro de tiempo de arquitectura antes de medirlo.
    if remaining < estimate*1.12 + 60:
        print(json.dumps(dict(status="pendiente_GPU", reason="No cabe otro run completo con margen", remaining=remaining, estimated=estimate)), flush=True)
        return
    if stats:
        cfg["STATS_PATH"] = stats
    cfg["DEADLINE_UNIX"] = state["deadline_unix"] - 30  # reserva para guardar y cerrar
    cfg["EXPERIMENT_DECISION"] = decision
    started = time.time()
    run, summary = train(cfg)
    state["completed"].append(dict(experiment=name, run=run.name, seconds=time.time()-started,
                                    decision=decision, status=summary["status"]))
    state["elapsed_seconds"] = time.time()-state["start_unix"]
    (directory / "budget.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    entries = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    for entry in entries:
        if entry["name"] == name:
            entry.update(status=summary["status"], run=run.name, decision=decision)
    (directory / "plan.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(json.dumps(dict(run=str(run), summary=summary, remaining_seconds=state["deadline_unix"]-time.time()), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--run")
    parser.add_argument("--decision", default="Baseline preespecificado")
    parser.add_argument("--stats")
    parser.add_argument("--benchmark-run", help="Medir una configuración concreta antes de decidir si cabe")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.session and args.benchmark_run:
        directory = project_path(args.session)
        state = json.loads((directory / "budget.json").read_text(encoding="utf-8"))
        if time.time() >= state["deadline_unix"]-60:
            raise RuntimeError("No queda presupuesto para otra medición")
        cfg = json.loads((directory / f"{args.benchmark_run}.json").read_text(encoding="utf-8"))
        measured = benchmark(cfg)
        with (directory / f"benchmark_{args.benchmark_run}.json").open("x", encoding="utf-8") as output:
            json.dump(measured, output, indent=2)
        print(json.dumps(measured, indent=2), flush=True)
    elif args.session and args.run:
        execute(args.session, args.run, args.decision, args.stats)
    else:
        parser.error("Usar --prepare o --session RUTA --run P3_01")


if __name__ == "__main__":
    main()
