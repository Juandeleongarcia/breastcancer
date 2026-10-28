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
                balanced_accuracy=float((recall + (tn / (tn + fp) if tn + fp else 0.0))/2),
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


def save_evaluation_artifacts(result, output_dir, predictions_dir=None, simple=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = Path(predictions_dir) if predictions_dir else output_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for level in ["slice", "patient"]:
        frame = result[f"{level}_predictions"]
        frame.to_csv(predictions_dir / f"predictions_{level}.csv", index=False)
        if frame.label.nunique() == 2:
            plot_roc_curve(frame.label, frame.probability, (output_dir / "roc_curve.png" if simple and level == "patient" else predictions_dir / f"roc_curve_{level}.png" if simple else output_dir / f"roc_curve_{level}.png"),
                           title="Validación interna por paciente · mejor checkpoint" if simple and level == "patient" else f"ROC por {level}")
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.asarray(result[level]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pCR", "pCR"],
               yticklabels=["no pCR", "pCR"], xlabel="Predicción", ylabel="Real", title=f"Matriz por {level}")
        fig.tight_layout()
        fig.savefig((predictions_dir if simple else output_dir) / f"confusion_matrix_{level}.png", dpi=150)
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

def plot_training_history(history_path="models/history.csv", output_dir="models"):
    """Paciente para accuracy/AUC; BCE por corte para loss, sin inventar históricos.

    Train es online con pesos cambiantes, dropout y augmentation si está activa.
    La validación usa pesos fijos al final de época, modo eval y sin augmentation.
    No son estimaciones equivalentes del mismo protocolo de evaluación.
    """
    history = pd.read_csv(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient = "val_patient_auc" in history
    for metric, ylabel, title in [
        ("loss", "BCE media por corte (ponderada si se configuró)", "Loss por corte"),
        ("accuracy", "Accuracy por paciente" if patient else "Accuracy por corte (histórico)", "Accuracy"),
        ("auc", "ROC AUC por paciente" if patient else "ROC AUC por corte (histórico)", "ROC AUC")]:
        level = "patient_" if patient and metric != "loss" else ""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        comparable = f"train_eval_{level}{metric}" in history
        if metric == "loss" or not comparable:
            ax.plot(history.epoch, history[f"train_{level}{metric}"], "o--", color="#777777",
                    label="Train online: pesos cambiantes; dropout/augment. según config")
        if comparable:
            ax.plot(history.epoch, history[f"train_eval_{level}{metric}"], "o-", color="#d17a00",
                    label="Train eval: fin de época, eval, sin augmentation")
        ax.plot(history.epoch, history[f"val_{level}{metric}"], "o-", linewidth=2.5, color="#1764ab",
                label="Validación interna: fin de época, eval, sin augmentation")
        if metric != "loss":
            ax.set_ylim(0, 1)
        if metric == "auc":
            ax.axhline(.5, color="black", linestyle=":", label="Azar: AUC = 0,50")
        for epoch, value in zip(history.epoch, history[f"val_{level}{metric}"]):
            if np.isfinite(value):
                ax.annotate(f"{value:.3f}", (epoch, value), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.set(xlabel="Época", ylabel=ylabel,
               title=title + (" · métrica principal por paciente" if metric == "auc" and patient else ""))
        ax.set_xticks(history.epoch)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc="best")
        note = ("Train eval y validación: pesos finales; BCE por corte / métricas por paciente."
                if comparable else "Train online y validación final usan protocolos distintos: interpretar la diferencia con cautela.")
        fig.text(.5, .015, note,
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(output_dir / f"{metric}_curve.png", dpi=160)
        plt.close(fig)

    if "train_eval_loss" in history:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for column, label, color in [
                ("train_eval_loss", "Train eval: fin de época, sin augmentation", "#d17a00"),
                ("val_loss", "Validación interna: fin de época, sin augmentation", "#1764ab")]:
            ax.plot(history.epoch, history[column], "o-", color=color, label=label)
        ax.set(xlabel="Época", ylabel="BCE media por corte (ponderada si se configuró)",
               title="Pérdidas comparables · mismos pesos finales y criterio")
        ax.set_xticks(history.epoch)
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_comparison.png", dpi=160)
        plt.close(fig)
