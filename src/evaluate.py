"""Evaluación independiente usando la configuración guardada con los pesos."""
import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.configuration import resolve_config, load_config, project_path, file_hash
from src.dataset import BreastDCEDataset
from src.model import BreastCancerCNN
from src.experiment import select_device, create_criterion, validate, select_patients
from src.metrics import save_evaluation_artifacts
from src.run_layout import run_root, technical_file


def evaluate_checkpoint(path, output_dir=None, legacy_config=None, csv_path=None):
    path = project_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw_config = checkpoint.get("config")
    if raw_config is None:
        candidates = [technical_file(run_root(path), "config.json"), path.parent / "config.json", path.parent / "details" / "config.json",
                      path.parent.parent / "details" / "config.json"]
        snapshot = next((p for p in candidates if p.exists()), candidates[0])
        if legacy_config:
            raw_config = load_config(legacy_config)
        elif snapshot.exists():
            raw_config = load_config(snapshot)
        else:
            raise ValueError("Checkpoint histórico sin contrato: aportar --legacy-config explícito")
        required = ["INPUT_CHANNELS", "CONV_CHANNELS", "KERNEL_SIZE", "PADDING", "POOL_SIZE",
                    "USE_BATCH_NORM", "DROPOUT", "IMAGE_SIZE", "INPUT_MODE", "USE_CLIPPING",
                    "CHANNEL_P01", "CHANNEL_P99", "USE_STANDARDIZATION", "CHANNEL_MEANS", "CHANNEL_STDS"]
        if any(key not in raw_config for key in required):
            raise ValueError("Contrato histórico incompleto: no se adivina el preprocesamiento")
    # Los checkpoints anteriores a la variante espacial usaban una convolución por bloque.
    # No dejar que una futura edición de defaults cambie su arquitectura al evaluarlos.
    raw_config = dict(raw_config)
    raw_config.setdefault("CONVS_PER_BLOCK", 1)
    cfg = resolve_config(raw_config)
    cfg["DEADLINE_UNIX"] = None  # El presupuesto original no limita una reevaluación posterior.
    selected_csv = csv_path or cfg["VAL_CSV"]
    expected_hash = checkpoint.get("csv_sha256", {}).get("validation")
    if csv_path is None and expected_hash and file_hash(selected_csv) != expected_hash:
        raise ValueError("La partición de validación cambió desde el entrenamiento")
    frame = pd.read_csv(project_path(selected_csv))
    if "split" not in frame or not frame.split.eq("train").all():
        raise ValueError("Este evaluador de desarrollo solo utiliza validación interna del train público")
    dataset = BreastDCEDataset(selected_csv, cfg["DATA_DIR"], config=cfg)
    if cfg["SMOKE_TEST"] and csv_path is None:
        dataset.df = select_patients(dataset.df, cfg["SMOKE_PATIENTS_PER_CLASS"], cfg["SEED"])
    device = select_device(cfg)
    model = BreastCancerCNN(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = create_criterion(cfg, device, checkpoint.get("pos_weight"))
    loader = DataLoader(dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                        num_workers=cfg["NUM_WORKERS"], pin_memory=device.type == "cuda")
    result = validate(model, loader, criterion, device, cfg)
    result["diagnostic_partial"] = bool(result["diagnostic_partial"] or cfg["SMOKE_TEST"])
    if output_dir is not None:
        output_dir = project_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        save_evaluation_artifacts(result, output_dir)
        summary = {k: result[k] for k in ["loss", "slice", "patient", "diagnostic_partial"]}
        (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluación reproducible por corte y paciente")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", help="Carpeta nueva, nunca sobrescrita")
    parser.add_argument("--legacy-config")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.output, args.legacy_config)
    print(json.dumps({k: result[k] for k in ["loss", "slice", "patient", "diagnostic_partial"]}, indent=2))


if __name__ == "__main__":
    main()
