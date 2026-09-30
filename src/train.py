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
    roc_auc_score
)

from torch.utils.data import DataLoader, Dataset

from dataset import BreastDCEDataset
from model import BreastCancerCNN


# =========================================================
# CONFIGURACIÓN
# =========================================================

SEED = 42

TRAIN_CSV = "metadata/internal_train.csv"
VAL_CSV = "metadata/internal_val.csv"

ROOT_DIR = "breastdcedl"

BATCH_SIZE = 8
LEARNING_RATE = 1e-3

# Para pruebas rápidas en portátil:
MAX_EPOCHS = 1

# Para entrenamiento serio luego:
# MAX_EPOCHS = 30

PATIENCE = 5

USE_WEIGHTED_LOSS = True

# ---------------------------------------------------------
# VARIANTE
#
# A = 2 convoluciones + 2 pooling
# B = 3 convoluciones + 3 pooling
# C = 3 convoluciones + 2 pooling
# ---------------------------------------------------------

MODEL_VARIANT = "A"


# =========================================================
# CARPETAS DE SALIDA
# =========================================================

MODEL_DIR = Path("models")

EXPERIMENT_DIR = (
    MODEL_DIR / f"model_{MODEL_VARIANT}"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    EXPERIMENT_DIR / "best.pt"
)

HISTORY_PATH = (
    EXPERIMENT_DIR / "history.csv"
)


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATASET INTERNO
# =========================================================

class InternalSplitDataset(Dataset):

    def __init__(self, csv_path, root_dir):

        self.df = pd.read_csv(csv_path)

        self.base_dataset = BreastDCEDataset(
            csv_path="metadata/samples.csv",
            root_dir=root_dir,
            split="train"
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
            dim=0
        )

        label = torch.tensor(
            float(row["pCR"]),
            dtype=torch.float32
        )

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"],
            "sample_id": row["sample_id"]
        }


# =========================================================
# POS WEIGHT
# =========================================================

def calculate_pos_weight(csv_path):

    df = pd.read_csv(csv_path)

    n0 = (df["pCR"] == 0).sum()
    n1 = (df["pCR"] == 1).sum()

    pos_weight = n0 / n1

    print("Muestras no pCR:", n0)
    print("Muestras pCR:", n1)
    print("pos_weight:", pos_weight)

    return pos_weight


# =========================================================
# TRAIN DE UNA ÉPOCA
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_probabilities = []

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(images).squeeze(1)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1 if p >= 0.5 else 0
        for p in all_probabilities
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    except ValueError:
        auc = float("nan")

    return (
        epoch_loss,
        accuracy,
        auc
    )


# =========================================================
# VALIDACIÓN
# =========================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1 if p >= 0.5 else 0
        for p in all_probabilities
    ]

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1]
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
        "specificity": specificity
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
    path
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "variant":
                MODEL_VARIANT,

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

            "max_epochs":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "weighted_loss":
                USE_WEIGHTED_LOSS
        },
        path
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("========================================")
    print("        ENTRENAMIENTO BREASTDCEDL")
    print("========================================")
    print()

    print("Modelo:", MODEL_VARIANT)
    print("Dispositivo:", device)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Máximo de épocas:", MAX_EPOCHS)
    print("Patience:", PATIENCE)
    print("Seed:", SEED)
    print()

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    train_dataset = InternalSplitDataset(
        csv_path=TRAIN_CSV,
        root_dir=ROOT_DIR
    )

    val_dataset = InternalSplitDataset(
        csv_path=VAL_CSV,
        root_dir=ROOT_DIR
    )

    print(
        "Muestras train:",
        len(train_dataset)
    )

    print(
        "Muestras validation:",
        len(val_dataset)
    )

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # MODELO
    # -----------------------------------------------------

    model = BreastCancerCNN(
        variant=MODEL_VARIANT
    )

    model = model.to(device)

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
            device=device
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

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    history = []

    total_start_time = time.time()

    # -----------------------------------------------------
    # LOOP PRINCIPAL
    # -----------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        epoch_start = time.time()

        (
            train_loss,
            train_acc,
            train_auc
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # -------------------------------------------------
        # MOSTRAR RESULTADOS
        # -------------------------------------------------

        print()
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train AUC: {train_auc:.4f}"
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

        # -------------------------------------------------
        # GUARDAR HISTORIAL
        # -------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_sensitivity":
                val_metrics["sensitivity"],
            "val_specificity":
                val_metrics["specificity"],
            "epoch_time_seconds":
                epoch_time
        }

        history.append(
            epoch_record
        )

        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False
        )

        # -------------------------------------------------
        # GUARDAR MODELO DE ESTA ÉPOCA
        # -------------------------------------------------

        checkpoint_path = (
            EXPERIMENT_DIR
            / f"epoch_{epoch:03d}.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            train_auc=train_auc,
            val_metrics=val_metrics,
            path=checkpoint_path
        )

        print(
            "Checkpoint guardado:",
            checkpoint_path
        )

        # -------------------------------------------------
        # GUARDAR MEJOR MODELO
        # -------------------------------------------------

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = (
                val_metrics["loss"]
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_auc=train_auc,
                val_metrics=val_metrics,
                path=BEST_MODEL_PATH
            )

            print(
                "Nuevo mejor modelo guardado:",
                BEST_MODEL_PATH
            )

        else:

            epochs_without_improvement += 1

            print(
                "Sin mejora:",
                epochs_without_improvement,
                "/",
                PATIENCE
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping activado."
            )

            break

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    total_time = (
        time.time()
        - total_start_time
    )

    print()
    print("========================================")
    print("       ENTRENAMIENTO FINALIZADO")
    print("========================================")
    print()

    print(
        "Modelo:",
        MODEL_VARIANT
    )

    print(
        "Mejor val loss:",
        best_val_loss
    )

    print(
        "Carpeta experimento:",
        EXPERIMENT_DIR
    )

    print(
        "Mejor modelo:",
        BEST_MODEL_PATH
    )

    print(
        "Historial:",
        HISTORY_PATH
    )

    print(
        f"Tiempo total: "
        f"{total_time:.2f} s"
    )


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# NOTA
#
# Cada entrenamiento guarda automáticamente:
#
#   models/model_A/epoch_001.pt
#   models/model_A/epoch_002.pt
#   ...
#   models/model_A/best.pt
#   models/model_A/history.csv
#
# y lo mismo para B y C.
#
# Cada checkpoint contiene:
#
#   - pesos de la CNN
#   - estado del optimizador
#   - arquitectura
#   - época
#   - train loss
#   - train accuracy
#   - train AUC
#   - validation loss
#   - validation accuracy
#   - validation AUC
#   - sensibilidad
#   - especificidad
#   - seed
#   - batch size
#   - learning rate
#   - configuración de la loss
#
# Así ningún modelo entrenado se pierde y se pueden comparar después todos
# los experimentos realizados.
# -------------------------------------------------------------------------