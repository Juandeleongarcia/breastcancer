from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import (
    VAL_CSV,
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASSIFICATION_THRESHOLD,
    MODEL_PATH,
    USE_WEIGHTED_LOSS,
    USE_GPU,
)

from src.model import BreastCancerCNN
from src.train import (
    InternalSplitDataset,
    calculate_pos_weight,
    validate,
)

from src.metrics import plot_roc_curve


def main():

    # =====================================================
    # DISPOSITIVO
    # =====================================================

    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("========================================")
    print("        EVALUACIÓN BREASTDCEDL")
    print("========================================")
    print()

    print("Dispositivo:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()

    # =====================================================
    # DATASET DE VALIDACIÓN
    # =====================================================

    val_dataset = InternalSplitDataset(
        csv_path=VAL_CSV,
        root_dir=DATA_DIR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(
        "Muestras validation:",
        len(val_dataset),
    )

    print()

    # =====================================================
    # MODELO
    # =====================================================

    model = BreastCancerCNN()

    model = model.to(device)

    # =====================================================
    # CARGAR CHECKPOINT
    # =====================================================

    model_path = Path(MODEL_PATH)

    if not model_path.exists():

        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    print(
        "Cargando modelo:",
        model_path,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Época guardada:",
        checkpoint.get("epoch", "desconocida"),
    )

    print(
        "Val AUC guardado:",
        checkpoint.get("val_auc", "desconocido"),
    )

    print()

    # =====================================================
    # LOSS
    # =====================================================

    if USE_WEIGHTED_LOSS:

        pos_weight_value = calculate_pos_weight(
            "metadata/internal_train.csv"
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

    else:

        criterion = nn.BCEWithLogitsLoss()

    # =====================================================
    # VALIDACIÓN
    # =====================================================

    val_metrics = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("========================================")
    print("          RESULTADOS VALIDACIÓN")
    print("========================================")
    print()

    print(
        f"Loss: "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{val_metrics['auc']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{val_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{val_metrics['specificity']:.4f}"
    )

    print(
        f"Threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    # =====================================================
    # CURVA ROC
    # =====================================================

    roc_path = (
        model_path.parent
        / "roc_curve.png"
    )

    plot_roc_curve(
        labels=val_metrics["labels"],
        probabilities=val_metrics["probabilities"],
        output_path=roc_path,
        title="ROC Curve - Validation",
    )

    print()
    print(
        "Curva ROC guardada en:",
        roc_path,
    )


if __name__ == "__main__":
    main()