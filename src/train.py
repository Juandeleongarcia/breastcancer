import random
import shutil
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

from torch.utils.data import DataLoader

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
    TARGET_ROC_AUC,
    SEED,
    USE_GPU,
    USE_WEIGHTED_LOSS,
)

from src.dataset import BreastDCEDataset
from src.model import BreastCancerCNN

from src.metrics import (
    plot_roc_curve,
    plot_training_history,
)


# =========================================================
# RUTAS
# =========================================================

MODELS_DIR = Path("models")
CONFIG_PATH = Path("config/config.py")

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CREACIÓN AUTOMÁTICA DE RUN
# =========================================================

def create_run_directory():

    existing_runs = []

    for path in MODELS_DIR.iterdir():

        if (
            path.is_dir()
            and path.name.startswith("run_")
        ):

            try:

                run_number = int(
                    path.name.split("_")[1]
                )

                existing_runs.append(
                    run_number
                )

            except (
                IndexError,
                ValueError,
            ):
                pass

    if existing_runs:

        next_number = (
            max(existing_runs) + 1
        )

    else:

        next_number = 1

    run_dir = (
        MODELS_DIR
        / f"run_{next_number:03d}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir


# =========================================================
# COPIA DEL CONFIG
# =========================================================

def copy_config(
    run_dir,
):

    shutil.copy2(
        CONFIG_PATH,
        run_dir / "config.py",
    )


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# PESO CLASE POSITIVA
# =========================================================

def calculate_pos_weight(
    csv_path,
):

    df = pd.read_csv(
        csv_path
    )

    n0 = (
        df["pCR"] == 0
    ).sum()

    n1 = (
        df["pCR"] == 1
    ).sum()

    if n1 == 0:

        raise ValueError(
            "No hay muestras positivas "
            "(pCR = 1) en entrenamiento."
        )

    pos_weight = (
        n0 / n1
    )

    print(
        "Muestras no pCR:",
        n0,
    )

    print(
        "Muestras pCR:",
        n1,
    )

    print(
        f"pos_weight: "
        f"{pos_weight:.4f}"
    )

    return pos_weight


# =========================================================
# TRAIN EPOCH
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

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["label"]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad()

        logits = (
            model(
                images
            )
            .squeeze(1)
        )

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

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1
        if probability
        >= CLASSIFICATION_THRESHOLD
        else 0
        for probability
        in all_probabilities
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

        auc = float(
            "nan"
        )

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

            images = (
                batch["image"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            labels = (
                batch["label"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            logits = (
                model(
                    images
                )
                .squeeze(1)
            )

            loss = criterion(
                logits,
                labels,
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    predictions = [
        1
        if probability
        >= CLASSIFICATION_THRESHOLD
        else 0
        for probability
        in all_probabilities
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

        auc = float(
            "nan"
        )

    tn, fp, fn, tp = (
        confusion_matrix(
            all_labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

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

        "loss":
            epoch_loss,

        "accuracy":
            accuracy,

        "auc":
            auc,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "labels":
            all_labels,

        "probabilities":
            all_probabilities,
    }


# =========================================================
# CHECKPOINT
# =========================================================

def save_epoch_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    train_auc,
    val_metrics,
    run_dir,
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

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

        "optimizer":
            OPTIMIZER,

        "epochs":
            EPOCHS,

        "patience":
            PATIENCE,

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "weighted_loss":
            USE_WEIGHTED_LOSS,
    }

    epoch_path = (
        run_dir
        / f"epoch_{epoch:03d}.pt"
    )

    torch.save(
        checkpoint,
        epoch_path,
    )

    return epoch_path


# =========================================================
# OPTIMIZADOR
# =========================================================

def create_optimizer(
    model,
):

    optimizer_name = (
        OPTIMIZER.lower()
    )

    if optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    raise ValueError(
        "Optimizador no reconocido: "
        f"{OPTIMIZER}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    run_dir = (
        create_run_directory()
    )

    history_path = (
        run_dir
        / "history.csv"
    )

    copy_config(
        run_dir
    )

    set_seed(
        SEED
    )

    # =====================================================
    # DISPOSITIVO
    # =====================================================

    if (
        USE_GPU
        and torch.cuda.is_available()
    ):

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print()

    print(
        "========================================"
    )

    print(
        "        ENTRENAMIENTO BREASTDCEDL"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Run:",
        run_dir,
    )

    print(
        "Dispositivo:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Optimizador:",
        OPTIMIZER,
    )

    print(
        "Máximo de épocas:",
        EPOCHS,
    )

    print(
        "Patience:",
        PATIENCE,
    )

    print(
        "Threshold:",
        CLASSIFICATION_THRESHOLD,
    )

    print(
        "Objetivo ROC-AUC:",
        TARGET_ROC_AUC,
    )

    print(
        "Seed:",
        SEED,
    )

    print()

    # =====================================================
    # DATASETS
    # =====================================================

    train_dataset = BreastDCEDataset(
        csv_path=TRAIN_CSV,
        root_dir=DATA_DIR,
        split=None,
    )

    val_dataset = BreastDCEDataset(
        csv_path=VAL_CSV,
        root_dir=DATA_DIR,
        split=None,
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

    # =====================================================
    # COMPROBACIÓN DE DATA LEAKAGE
    # =====================================================

    train_patients = set(
        train_dataset.df[
            "patient_id"
        ]
        .astype(str)
    )

    val_patients = set(
        val_dataset.df[
            "patient_id"
        ]
        .astype(str)
    )

    common_patients = (
        train_patients
        & val_patients
    )

    if common_patients:

        raise ValueError(
            "DATA LEAKAGE DETECTADO: "
            f"{len(common_patients)} pacientes "
            "están en train y validation."
        )

    print(
        "Comprobación paciente train/val: OK"
    )

    print(
        "Pacientes train:",
        len(train_patients),
    )

    print(
        "Pacientes validation:",
        len(val_patients),
    )

    print()

    # =====================================================
    # DATALOADERS
    # =====================================================

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

    # =====================================================
    # MODELO
    # =====================================================

    model = (
        BreastCancerCNN()
        .to(
            device
        )
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
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

    # =====================================================
    # LOSS
    # =====================================================

    if USE_WEIGHTED_LOSS:

        pos_weight_value = (
            calculate_pos_weight(
                TRAIN_CSV
            )
        )

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            )
        )

        print(
            "Loss: BCEWithLogitsLoss ponderada"
        )

    else:

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        print(
            "Loss: BCEWithLogitsLoss normal"
        )

    print()

    optimizer = (
        create_optimizer(
            model
        )
    )

    # =====================================================
    # CONTROL
    # =====================================================

    best_val_auc = float(
        "-inf"
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    total_start_time = (
        time.time()
    )

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )

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

        # =================================================
        # HISTORY
        # =================================================

        history.append(
            {

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
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # =================================================
        # CHECKPOINT
        # =================================================

        epoch_path = (
            save_epoch_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_auc=train_auc,
                val_metrics=val_metrics,
                run_dir=run_dir,
            )
        )

        print(
            "Checkpoint guardado:",
            epoch_path,
        )

        # =================================================
        # BEST AUC
        # =================================================

        current_auc = (
            val_metrics["auc"]
        )

        auc_improved = (
            not np.isnan(
                current_auc
            )
            and
            current_auc
            > best_val_auc
        )

        if auc_improved:

            best_val_auc = (
                current_auc
            )

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            print(
                "Nueva mejor época:",
                best_epoch,
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

        if (
            not np.isnan(
                current_auc
            )
            and current_auc
            >= TARGET_ROC_AUC
        ):

            print(
                f"*** OBJETIVO ROC-AUC "
                f"{TARGET_ROC_AUC:.2f} "
                f"SUPERADO ***"
            )

        if (
            USE_EARLY_STOPPING
            and
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping activado."
            )

            break

    # =====================================================
    # MEJOR ÉPOCA
    # =====================================================

    if best_epoch == 0:

        raise RuntimeError(
            "No se pudo determinar una mejor época."
        )

    best_epoch_path = (
        run_dir
        / f"epoch_{best_epoch:03d}.pt"
    )

    print()

    print(
        "Cargando mejor época:",
        best_epoch_path,
    )

    checkpoint = torch.load(
        best_epoch_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_val_metrics = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # =====================================================
    # ROC
    # =====================================================

    roc_path = (
        run_dir
        / "roc_curve.png"
    )

    plot_roc_curve(
        labels=
            best_val_metrics[
                "labels"
            ],

        probabilities=
            best_val_metrics[
                "probabilities"
            ],

        output_path=
            roc_path,

        title=
            "ROC Curve - Best Validation Epoch",
    )

    # =====================================================
    # CURVAS
    # =====================================================

    plot_training_history(
        history_path=
            history_path,

        output_dir=
            run_dir,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_path = (
        run_dir
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Best epoch: "
            f"{best_epoch}\n"
        )

        file.write(
            f"Best validation AUC: "
            f"{best_val_auc:.6f}\n"
        )

        file.write(
            f"Validation loss: "
            f"{best_val_metrics['loss']:.6f}\n"
        )

        file.write(
            f"Validation accuracy: "
            f"{best_val_metrics['accuracy']:.6f}\n"
        )

        file.write(
            f"Validation sensitivity: "
            f"{best_val_metrics['sensitivity']:.6f}\n"
        )

        file.write(
            f"Validation specificity: "
            f"{best_val_metrics['specificity']:.6f}\n"
        )

        file.write(
            f"Target AUC: "
            f"{TARGET_ROC_AUC:.6f}\n"
        )

        file.write(
            f"Optimizer: "
            f"{OPTIMIZER}\n"
        )

        file.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        file.write(
            f"Weight decay: "
            f"{WEIGHT_DECAY}\n"
        )

        file.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        file.write(
            f"Patience: "
            f"{PATIENCE}\n"
        )

    # =====================================================
    # FINAL
    # =====================================================

    total_time = (
        time.time()
        - total_start_time
    )

    print()

    print(
        "========================================"
    )

    print(
        "        ENTRENAMIENTO FINALIZADO"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Run guardado en:",
        run_dir,
    )

    print(
        "Mejor época:",
        best_epoch,
    )

    print(
        "Mejor checkpoint:",
        best_epoch_path,
    )

    print(
        f"Mejor Val AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Val Loss mejor época: "
        f"{best_val_loss:.4f}"
    )

    print(
        "Objetivo ROC-AUC:",
        TARGET_ROC_AUC,
    )

    if (
        best_val_auc
        >= TARGET_ROC_AUC
    ):

        print(
            "Resultado: "
            "OBJETIVO SUPERADO"
        )

    else:

        print(
            "Resultado: "
            "objetivo todavía no alcanzado"
        )

    print()

    print(
        "Archivos generados:"
    )

    print(
        "- config.py"
    )

    print(
        "- epoch_XXX.pt"
    )

    print(
        "- history.csv"
    )

    print(
        "- roc_curve.png"
    )

    print(
        "- loss_curve.png"
    )

    print(
        "- accuracy_curve.png"
    )

    print(
        "- auc_curve.png"
    )

    print(
        "- summary.txt"
    )

    print()

    print(
        f"Tiempo total: "
        f"{total_time:.2f} s"
    )


if __name__ == "__main__":

    main()