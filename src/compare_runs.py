"""Comparación automática: solo valores guardados, pendientes sin métricas inventadas."""
import argparse
from datetime import datetime
import json
from uuid import uuid4

import pandas as pd

from src.configuration import project_path


def comparison(root="models", phase3_session=None):
    rows = []
    for path in sorted(project_path(root).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "validation" not in summary:
            continue  # Batch fijo: no es un experimento con validación.
        val = summary["validation"]
        config_paths = [path.parent / "details" / "config.json", path.parent / "config.json"]
        config_path = next((p for p in config_paths if p.exists()), None)
        cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
        rows.append(dict(Run=summary["run"], Experimento=summary["experiment"],
                         Referencia=summary.get('comparison_parent',cfg.get('COMPARISON_PARENT')),
                         CNN=str(summary["cnn"]), Loss="ponderada" if summary["weighted_loss"] else "normal",
                         Clipping=summary["clipping"], Epocas=summary["epochs_completed"],
                         Mejor_epoca=summary["best_epoch"], ROC_AUC_paciente=val["patient"]["auc"],
                         ROC_AUC_corte=val["slice"]["auc"], Accuracy_paciente=val["patient"]["accuracy"],
                         Recall_paciente=val["patient"]["recall"], F1_paciente=val["patient"]["f1"],
                         Especificidad_paciente=val['patient']['specificity'],
                         Precision_paciente=val["patient"]["precision"],
                         LR=cfg.get("LEARNING_RATE"), Dropout=cfg.get("DROPOUT"),
                         Convoluciones_por_bloque=cfg.get("CONVS_PER_BLOCK", 1),
                         Scheduler=cfg.get("SCHEDULER", "none"),
                         Augmentation=cfg.get('AUGMENTATION_MODE', 'legacy_flips') if cfg.get('USE_AUGMENTATION') else 'none',
                         Interpretacion=summary.get("interpretation", "historico_exploratorio"),
                         Representacion=cfg.get("INPUT_MODE"), Normalizacion=cfg.get("USE_STANDARDIZATION"),
                         Train_pacientes=summary.get("train_patients"), Val_pacientes=summary.get("validation_patients"),
                         Grupo_comparable="preliminar" if summary["diagnostic_partial"] else "train_completo",
                         Val_loss=val["loss"], Tiempo_s=summary["seconds"], Estado=summary["status"]))
    for name in "ABCD" if phase3_session is None else []:
        if not any(row["Experimento"] == name and row["Estado"] == "entrenamiento_completo" for row in rows):
            rows.append(dict(Run=f"{name} completo", Experimento=name, Estado="experimento_pendiente"))
    if phase3_session:
        entries = json.loads((project_path(phase3_session) / "plan.json").read_text(encoding="utf-8"))
        names = {entry["name"] for entry in entries}
        rows = [row for row in rows if row["Experimento"] in names]
        executed = {row["Experimento"] for row in rows}
        for entry in entries:
            if entry["name"] not in executed:
                cfg = entry["config"]
                rows.append(dict(Run=entry["name"], Experimento=entry["name"], CNN=str(cfg["CONV_CHANNELS"]),
                    Loss="ponderada" if cfg["USE_WEIGHTED_LOSS"] else "normal", Clipping=cfg["USE_CLIPPING"],
                    LR=cfg["LEARNING_RATE"], Dropout=cfg["DROPOUT"], Representacion=cfg["INPUT_MODE"],
                    Normalizacion=cfg["USE_STANDARDIZATION"], Epocas_previstas=cfg["EPOCHS"], Estado="pendiente_GPU"))
    else:
        rows.append(dict(Run="Runs históricos del ordenador de clase", Estado="historico_no_disponible_localmente"))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="models")
    parser.add_argument("--output")
    parser.add_argument("--phase3-session")
    args = parser.parse_args()
    frame = comparison(args.root, args.phase3_session)
    path = project_path(args.output) if args.output else project_path(args.root) / (
        f"comparison_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Tabla guardada en: {path}")


if __name__ == "__main__":
    main()
