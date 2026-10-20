"""Entrada histórica del entrenamiento; lógica compartida en src.experiment.

Se conservan los helpers públicos y el formato de checkpoints por época.
Los nuevos runs añaden configuración completa y selección por AUC de paciente.
"""
import shutil
from pathlib import Path
import torch
from config.config import (
    SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER, EPOCHS,
    PATIENCE, CLASSIFICATION_THRESHOLD, USE_WEIGHTED_LOSS,
)
from src.experiment import (
    main, train, validate, create_optimizer, create_run_directory, calculate_pos_weight,
    set_seed, run_epoch,
)
from src.configuration import project_path

MODELS_DIR = Path("models")
CONFIG_PATH = Path("config/config.py")


def copy_config(run_dir):
    shutil.copy2(project_path(CONFIG_PATH), Path(run_dir) / "config.py")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Retorno histórico: loss, accuracy y AUC por corte."""
    result = run_epoch(model, loader, criterion, device, optimizer=optimizer,
                       threshold=CLASSIFICATION_THRESHOLD)
    return result["loss"], result["slice"]["accuracy"], result["slice"]["auc"]


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

    # El helper histórico también conserva el contrato del modelo actualizado.
    if hasattr(model, "config"):
        checkpoint["config"] = model.config
    if "patient" in val_metrics:
        checkpoint["val_patient_auc"] = val_metrics["patient"]["auc"]

    epoch_path = (
        run_dir
        / f"epoch_{epoch:03d}.pt"
    )

    if epoch_path.exists():
        raise FileExistsError(f"No se sobrescribe un checkpoint existente: {epoch_path}")

    torch.save(
        checkpoint,
        epoch_path,
    )

    return epoch_path



if __name__ == "__main__":
    import argparse
    from src.resume_validation import validate_resume_environment
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume")
    resume_args, _ = parser.parse_known_args()
    if resume_args.resume:
        validate_resume_environment(project_path(resume_args.resume))
    main()
