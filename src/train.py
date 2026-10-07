import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from torch.utils.data import DataLoader, Dataset

from config.config import (
    TRAIN_CSV,
    VAL_CSV,
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    OPTIMIZER,
    CLASSIFICATION_THRESHOLD,
    USE_EARLY_STOPPING,
    PATIENCE,
    MODEL_PATH,
    SAVE_BEST_MODEL,
    TARGET_ROC_AUC,
    SEED,
    USE_GPU,
    USE_WEIGHTED_LOSS,
)

from src.dataset import BreastDCEDataset
from src.model import BreastCancerCNN
from src.metrics import plot_roc_curve


# =========================================================
# RUTAS DE SALIDA
# =========================================================

MODEL_PATH = Path(MODEL_PATH)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

HISTORY_PATH = MODEL_PATH.parent / "history.csv"


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# DATASET PARA SPLIT INTERNO
# =========================================================

class InternalSplitDataset(Dataset):

    def __init__(
        self,
        csv_path,
        root_dir,
    ):
        self.df = pd.read_csv(csv_path)

        self.base_dataset = BreastDCEDataset(
            csv_path="metadata/samples.csv",
            root_dir=root_dir,
            split="train",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pre = self.base_dataset._load_grayscale_image(
            row["path_pre"]
        )

        early = self.base_dataset._load_grayscale_image(
            row["path_early"]
        )

        late = self.base_dataset._load_grayscale_image(
            row["path_late"]
        )

        image = torch.stack(
            [pre, early, late],
            dim=0,
        )

        label = torch.tensor(
            float(row["pCR"]),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"],
            "sample_id": row["sample_id"],
        }


# =========================================================
# PESO PARA CLASE POSITIVA
# =========================================================

def calculate_pos_weight(csv_path):

    df = pd.read_csv(csv_path)

    n0 = (df["pCR"] == 0).sum()
    n1 = (df["pCR"] == 1).sum()

    if n1 == 0:
        raise ValueError(
            "No hay muestras positivas (pCR = 1) "
            "en el conjunto de entrenamiento."
        )

    pos_weight = n0 / n1

    print("Muestras no pCR:", n0)
    print("Muestras pCR:", n1)
    print(f"pos_weight: {pos_weight:.4f}")

    return pos_weight


# =========================================================
# ENTRENAMIENTO DE UNA ÉPOCA
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_probabilities = []

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits = model(images).squeeze(1)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(logits)

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1 if probability >= CLASSIFICATION_THRESHOLD else 0
        for probability in all_probabilities
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions,
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities,
        )

    except ValueError:
        auc = float("nan")

    return (
        epoch_loss,
        accuracy,
        auc,
    )


# =========================================================
# VALIDACIÓN
# =========================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            logits = model(images).squeeze(1)

            loss = criterion(
                logits,
                labels,
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.sigmoid(logits)

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1 if probability >= CLASSIFICATION_THRESHOLD else 0
        for probability in all_probabilities
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions,
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities,
        )

    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "labels": all_labels,
        "probabilities": all_probabilities,
    }


# =========================================================
# GUARDAR CHECKPOINT
# =========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    train_auc,
    val_metrics,
    path,
):

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "train_auc":
            train_auc,

        "val_loss":
            val_metrics["loss"],

        "val_accuracy":
            val_metrics["accuracy"],

        "val_auc":
            val_metrics["auc"],

        "val_sensitivity":
            val_metrics["sensitivity"],

        "val_specificity":
            val_metrics["specificity"],

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "epochs":
            EPOCHS,

        "patience":
            PATIENCE,

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "weighted_loss":
            USE_WEIGHTED_LOSS,
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# OPTIMIZADOR
# =========================================================

def create_optimizer(model):

    optimizer_name = OPTIMIZER.lower()

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Optimizador no reconocido: {OPTIMIZER}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    # -----------------------------------------------------
    # DISPOSITIVO
    # -----------------------------------------------------

    if USE_GPU and torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print()
    print("========================================")
    print("        ENTRENAMIENTO BREASTDCEDL")
    print("========================================")
    print()

    print("Dispositivo:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    elif USE_GPU:

        print(
            "AVISO: USE_GPU=True, "
            "pero CUDA no está disponible."
        )

    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Optimizador:", OPTIMIZER)
    print("Máximo de épocas:", EPOCHS)
    print("Patience:", PATIENCE)
    print("Umbral:", CLASSIFICATION_THRESHOLD)
    print("Objetivo ROC-AUC:", TARGET_ROC_AUC)
    print("Seed:", SEED)
    print()

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    train_dataset = InternalSplitDataset(
        csv_path=TRAIN_CSV,
        root_dir=DATA_DIR,
    )

    val_dataset = InternalSplitDataset(
        csv_path=VAL_CSV,
        root_dir=DATA_DIR,
    )

    print(
        "Muestras train:",
        len(train_dataset),
    )

    print(
        "Muestras validation:",
        len(val_dataset),
    )

    print()

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # -----------------------------------------------------
    # MODELO
    # -----------------------------------------------------

    model = BreastCancerCNN()

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Parámetros totales:",
        f"{total_parameters:,}",
    )

    print(
        "Parámetros entrenables:",
        f"{trainable_parameters:,}",
    )

    print()

    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    if USE_WEIGHTED_LOSS:

        pos_weight_value = calculate_pos_weight(
            TRAIN_CSV
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        print(
            "Loss: BCEWithLogitsLoss ponderada"
        )

    else:

        criterion = nn.BCEWithLogitsLoss()

        print(
            "Loss: BCEWithLogitsLoss normal"
        )

    print()

    # -----------------------------------------------------
    # OPTIMIZADOR
    # -----------------------------------------------------

    optimizer = create_optimizer(model)

    # -----------------------------------------------------
    # CONTROL DEL MEJOR MODELO
    # -----------------------------------------------------

    best_val_auc = float("-inf")

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    total_start_time = time.time()

    # -----------------------------------------------------
    # LOOP PRINCIPAL
    # -----------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        (
            train_loss,
            train_acc,
            train_auc,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Train AUC: "
            f"{train_auc:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val AUC: "
            f"{val_metrics['auc']:.4f}"
        )

        print(
            f"Val Sensitivity: "
            f"{val_metrics['sensitivity']:.4f}"
        )

        print(
            f"Val Specificity: "
            f"{val_metrics['specificity']:.4f}"
        )

        print(
            f"Tiempo época: "
            f"{epoch_time:.2f} s"
        )

        if (
            not np.isnan(val_metrics["auc"])
            and val_metrics["auc"] >= TARGET_ROC_AUC
        ):

            print(
                f"*** Objetivo ROC-AUC "
                f"{TARGET_ROC_AUC:.2f} "
                f"SUPERADO ***"
            )

        # -------------------------------------------------
        # HISTORIAL
        # -------------------------------------------------

        epoch_record = {

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "train_auc":
                train_auc,

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_auc":
                val_metrics["auc"],

            "val_sensitivity":
                val_metrics["sensitivity"],

            "val_specificity":
                val_metrics["specificity"],

            "epoch_time_seconds":
                epoch_time,
        }

        history.append(
            epoch_record
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False,
        )

        # -------------------------------------------------
        # MEJOR MODELO SEGÚN ROC-AUC
        # -------------------------------------------------

        current_auc = val_metrics["auc"]

        auc_improved = (
            not np.isnan(current_auc)
            and current_auc > best_val_auc
        )

        if auc_improved:

            best_val_auc = current_auc

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            if SAVE_BEST_MODEL:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    train_auc=train_auc,
                    val_metrics=val_metrics,
                    path=MODEL_PATH,
                )

                print(
                    "Nuevo mejor modelo guardado:",
                    MODEL_PATH,
                )

                print(
                    f"Mejor Val AUC: "
                    f"{best_val_auc:.4f}"
                )

        else:

            epochs_without_improvement += 1

            print(
                "Sin mejora del AUC:",
                epochs_without_improvement,
                "/",
                PATIENCE,
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if (
            USE_EARLY_STOPPING
            and epochs_without_improvement >= PATIENCE
        ):

            print()
            print(
                "Early stopping activado."
            )

            break

    # =====================================================
    # CURVA ROC DEL MEJOR MODELO
    # =====================================================

    if MODEL_PATH.exists():

        print()
        print(
            "Cargando mejor modelo para calcular ROC..."
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        best_val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        roc_path = (
            MODEL_PATH.parent
            / "roc_curve.png"
        )

        plot_roc_curve(
            labels=best_val_metrics["labels"],
            probabilities=best_val_metrics["probabilities"],
            output_path=roc_path,
            title="ROC Curve - Best Validation Model",
        )

    else:

        print(
            "No se encontró un modelo guardado "
            "para generar la curva ROC."
        )

    # =====================================================
    # FINAL
    # =====================================================

    total_time = (
        time.time()
        - total_start_time
    )

    print()
    print("========================================")
    print("        ENTRENAMIENTO FINALIZADO")
    print("========================================")
    print()

    print(
        "Mejor época:",
        best_epoch,
    )

    print(
        "Mejor Val AUC:",
        f"{best_val_auc:.4f}",
    )

    print(
        "Val Loss de la mejor época:",
        f"{best_val_loss:.4f}",
    )

    print(
        "Objetivo ROC-AUC:",
        TARGET_ROC_AUC,
    )

    if best_val_auc >= TARGET_ROC_AUC:

        print(
            "Resultado: OBJETIVO SUPERADO"
        )

    else:

        print(
            "Resultado: objetivo todavía no alcanzado"
        )

    print(
        "Mejor modelo:",
        MODEL_PATH,
    )

    print(
        "Historial:",
        HISTORY_PATH,
    )

    print(
        "Curva ROC:",
        MODEL_PATH.parent / "roc_curve.png",
    )

    print(
        f"Tiempo total: "
        f"{total_time:.2f} s"
    )


if __name__ == "__main__":
    main()