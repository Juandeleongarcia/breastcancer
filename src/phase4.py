"""Campaña CPU de dos épocas por run, selección manual informada y límite 90 min."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd

from src.batch_train import resolve_job
from src.configuration import project_path, load_config
from src.experiment import train, TrainingBudgetExceeded
from src.phase3 import benchmark
from src.compare_runs import comparison


def prepare():
    directory = project_path("models") / f"PHASE4_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:12]}"
    directory.mkdir(exist_ok=False)
    started = time.time()
    budget = dict(start_unix=started, deadline_unix=started+5400, budget_minutes=90, completed=[])
    (directory/"budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    suite = load_config("config/phase4/today_suite.json")
    (directory/"plan.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    manifest = {str(p.relative_to(project_path("models"))): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in project_path("models").glob("RUN_*/*") if p.is_file()}
    # Incluye también pesos/detalles históricos; no se escriben en esos directorios.
    for run in project_path("models").glob("RUN_*"):
        for p in run.rglob("*"):
            if p.is_file():
                manifest[str(p.relative_to(project_path("models")))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (directory/"historical_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg = resolve_job(suite["experiments"][0]["config"], "cpu")
    measured = benchmark(cfg)
    measured["estimated_two_epoch_run_seconds"] = measured["estimated_epoch_seconds"]*2+20
    (directory/"benchmark.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
    print(json.dumps(dict(session=str(directory), benchmark=measured), indent=2), flush=True)


def execute(session, experiment, decision):
    directory = project_path(session)
    budget = load_config(directory/"budget.json")
    suite = load_config(directory/"plan.json")
    entry = next(x for x in suite["experiments"] if x["id"] == experiment)
    if any(item["experiment"] == experiment for item in budget["completed"]):
        raise ValueError("Este experimento ya tiene un run; no se repite ni sobrescribe")
    cfg = resolve_job(entry["config"], "cpu")
    if budget["deadline_unix"]-time.time() < 120:
        print(json.dumps(dict(status="pendiente_GPU", experiment=experiment, reason="Sin margen de presupuesto")), flush=True)
        return
    if cfg["EPOCHS"] != 2 or cfg["SMOKE_TEST"] or cfg.get("TRAIN_PATIENTS_PER_CLASS") or cfg["MAX_TRAIN_BATCHES"] or cfg["MAX_VAL_BATCHES"]:
        raise ValueError("Hoy solo dos épocas completas, sin subconjuntos ni límites de batches")
    for key, count in [("TRAIN_CSV", 877), ("VAL_CSV", 220)]:
        if pd.read_csv(project_path(cfg[key])).patient_id.nunique() != count:
            raise ValueError("Las particiones no tienen el número de pacientes autorizado")
    estimate = load_config(directory/"benchmark.json")["estimated_two_epoch_run_seconds"]
    if budget["completed"]:
        estimate = max(item["seconds"] for item in budget["completed"])
    specific = directory/f"benchmark_{experiment}.json"
    if cfg["CONVS_PER_BLOCK"] > 1:
        try:
            measured = benchmark(cfg, deadline=budget["deadline_unix"]-30)
        except TrainingBudgetExceeded:
            print(json.dumps(dict(status="pendiente_GPU", experiment=experiment, reason="Presupuesto durante benchmark")), flush=True)
            return
        measured["estimated_two_epoch_run_seconds"] = measured["estimated_epoch_seconds"]*2+20
        with specific.open("x", encoding="utf-8") as output:
            json.dump(measured, output, indent=2)
        estimate = measured["estimated_two_epoch_run_seconds"]
    remaining = budget["deadline_unix"]-time.time()
    if remaining < estimate*1.10+45:
        print(json.dumps(dict(status="pendiente_GPU", experiment=experiment, remaining=remaining, estimate=estimate)), flush=True)
        return
    cfg.update(DEADLINE_UNIX=budget["deadline_unix"]-30, EXPERIMENT_DECISION=decision)
    started = time.time()
    run, summary = train(cfg)
    budget["completed"].append(dict(experiment=experiment, run=run.name, seconds=time.time()-started,
                                     decision=decision, status=summary["status"]))
    budget["elapsed_seconds"] = time.time()-budget["start_unix"]
    (directory/"budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    frame = comparison()
    names = {item["run"] for item in budget["completed"]}
    frame[frame.Run.isin(names)].to_csv(directory/"comparison.csv", index=False)
    print(json.dumps(dict(run=str(run), summary=summary, remaining_seconds=budget["deadline_unix"]-time.time()), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--run")
    parser.add_argument("--decision", default="Comparación prioritaria preespecificada")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.session and args.run:
        execute(args.session, args.run, args.decision)
    else:
        parser.error("Usar --prepare o --session CARPETA --run IDENTIFICADOR")


if __name__ == "__main__":
    main()
