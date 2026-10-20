from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)


def binary_metrics(labels, probabilities, threshold=0.5):
    """AUC continuo; matriz siempre en orden [[TN, FP], [FN, TP]]."""
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if not len(y) or y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Etiquetas y puntuaciones deben ser vectores no vacíos del mismo tamaño")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Etiquetas o probabilidades inválidas")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = float(recall_score(y, pred, zero_division=0))
    return dict(auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=recall,
                f1=float(f1_score(y, pred, zero_division=0)), sensitivity=recall,
                specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], n=len(y))


def evaluate_predictions(rows, threshold=0.5):
    """Agrupa por identificador explícito, nunca por posiciones de batch."""
    frame = pd.DataFrame(rows).copy()
    required = ["sample_id", "patient_id", "label", "probability"]
    if frame.empty or any(k not in frame for k in required) or frame[required].isna().any().any():
        raise ValueError("Predicciones vacías o identificadores/valores ausentes")
    if frame.sample_id.duplicated().any():
        raise ValueError("Hay predicciones duplicadas para un mismo sample_id")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("Etiquetas contradictorias dentro de una paciente")
    slices = binary_metrics(frame.label, frame.probability, threshold)
    patients = frame.groupby("patient_id", sort=True).agg(
        label=("label", "first"), probability=("probability", "mean"), n_slices=("sample_id", "size")
    ).reset_index()
    patients["prediction"] = (patients.probability >= threshold).astype(int)
    frame["prediction"] = (frame.probability >= threshold).astype(int)
    return {"slice": slices, "patient": binary_metrics(patients.label, patients.probability, threshold),
            "slice_predictions": frame, "patient_predictions": patients}


def save_evaluation_artifacts(result, output_dir, predictions_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = Path(predictions_dir) if predictions_dir else output_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for level in ["slice", "patient"]:
        frame = result[f"{level}_predictions"]
        frame.to_csv(predictions_dir / f"predictions_{level}.csv", index=False)
        if frame.label.nunique() == 2:
            plot_roc_curve(frame.label, frame.probability, output_dir / f"roc_curve_{level}.png",
                           title=f"ROC por {level}")
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.asarray(result[level]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pCR", "pCR"],
               yticklabels=["no pCR", "pCR"], xlabel="Predicción", ylabel="Real", title=f"Matriz por {level}")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{level}.png", dpi=150)
        plt.close(fig)


def analyze_probabilities(result, output_dir, details_dir=None):
    """Distribuciones y errores descriptivos, siempre al umbral ya fijado."""
    import json
    output_dir = Path(output_dir)
    details = Path(details_dir) if details_dir else output_dir
    details.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for level in ["patient", "slice"]:
        frame = result[f"{level}_predictions"].copy()
        groups = {}
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, name in [(0, "no pCR"), (1, "pCR")]:
            probabilities = frame.loc[frame.label == label, "probability"]
            groups[str(label)] = dict(n=len(probabilities), mean=float(probabilities.mean()),
                                      std=float(probabilities.std(ddof=0)),
                                      quantiles={str(q): float(probabilities.quantile(q)) for q in [0, .1, .5, .9, 1]})
            ax.hist(probabilities, bins=np.linspace(0, 1, 26), alpha=.55, label=name)
        ax.set(xlabel="Puntuación pCR", ylabel="Número de observaciones", title=f"Probabilidades por {level}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"probability_distribution_{level}.png", dpi=150)
        plt.close(fig)
        frame["absolute_error"] = (frame.label - frame.probability).abs()
        errors = frame[frame.label != frame.prediction].sort_values("absolute_error", ascending=False)
        errors.to_csv(details / f"errors_{level}.csv", index=False)
        diagnostics[level] = dict(by_label=groups, predicted_positive_fraction=float(frame.prediction.mean()),
                                  n_errors=len(errors), metrics=result[level])
    (details / "probability_analysis.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def plot_roc_curve(
    labels,
    probabilities,
    output_path="models/roc_curve.png",
    title="ROC Curve - Validation",
):
    """
    Calcula y guarda la curva ROC.

    Parameters
    ----------
    labels : list or array
        Etiquetas reales (0 o 1).

    probabilities : list or array
        Probabilidades predichas por el modelo.

    output_path : str
        Ruta donde se guardará la imagen.

    title : str
        Título de la gráfica.

    Returns
    -------
    roc_auc : float
        Área bajo la curva ROC.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    # Calculamos los puntos de la curva ROC
    fpr, tpr, thresholds = roc_curve(
        labels,
        probabilities,
    )

    # Calculamos el área bajo la curva
    roc_auc = auc(
        fpr,
        tpr,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Crear gráfica
    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )

    # Línea de referencia:
    # comportamiento equivalente al azar
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier (AUC = 0.50)",
    )

    plt.xlabel(
        "False Positive Rate (1 - Specificity)"
    )

    plt.ylabel(
        "True Positive Rate (Sensitivity)"
    )

    plt.title(title)

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Curva ROC guardada en: {output_path}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    return roc_auc


def calculate_confusion_metrics(
    labels,
    probabilities,
    threshold=0.5,
):
    """
    Calcula métricas derivadas de la matriz de confusión.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value":
            negative_predictive_value,
    }

def plot_training_history(
    history_path="models/history.csv",
    output_dir="models",
):
    """
    Genera las curvas de entrenamiento:
    - Train Loss vs Validation Loss
    - Train Accuracy vs Validation Accuracy
    - Train AUC vs Validation AUC
    """

    import pandas as pd

    history = pd.read_csv(history_path)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # LOSS
    # =====================================================

    plt.figure(figsize=(7, 6))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # ACCURACY
    # =====================================================

    plt.figure(figsize=(7, 6))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.ylim(0, 1)
    plt.xticks(history["epoch"])

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # =====================================================
    # AUC
    # =====================================================

    plt.figure(figsize=(7, 6))

    plt.plot(
        history["epoch"],
        history["train_auc"],
        label="Train AUC",
    )

    plt.plot(
        history["epoch"],
        history["val_auc"],
        label="Validation AUC",
    )

    plt.ylim(0, 1)
    plt.xticks(history["epoch"])

    plt.axhline(
        y=0.70,
        linestyle="--",
        label="Target AUC = 0.70",
    )

    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Training and Validation ROC-AUC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "auc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Curvas de entrenamiento guardadas en:",
        output_dir,
    )

    # Nombres nuevos sin perder los nombres históricos de las gráficas.
    import shutil
    shutil.copy2(output_dir / "loss_curve.png", output_dir / "loss_history.png")
    if "val_patient_auc" in history:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(history.epoch, history.train_patient_auc, label="Train online, paciente")
        ax.plot(history.epoch, history.val_patient_auc, label="Validación, paciente")
        ax.set(xlabel="Época", ylabel="ROC AUC por paciente")
        ax.set_ylim(0, 1)
        ax.set_xticks(history.epoch)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "auc_history.png", dpi=150)
        plt.close(fig)
        for metric in ["accuracy", "auc"]:
            for level in ["patient", "slice"]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(history.epoch, history[f"train_{level}_{metric}"], marker="o", label="Train online")
                ax.plot(history.epoch, history[f"val_{level}_{metric}"], marker="o", label="Validación")
                label = "paciente" if level == "patient" else "corte"
                ax.set(xlabel="Época", ylabel=metric, title=f"{metric} por {label}", ylim=(0, 1))
                ax.set_xticks(history.epoch)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"{metric}_curve_{level}.png", dpi=150)
                plt.close(fig)
        if "learning_rate" in history:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(history.epoch, history.learning_rate, marker="o")
            ax.set(xlabel="Época", ylabel="Learning rate", yscale="log")
            ax.set_xticks(history.epoch)
            fig.tight_layout()
            fig.savefig(output_dir / "learning_rate_curve.png", dpi=150)
            plt.close(fig)
