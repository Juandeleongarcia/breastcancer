from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)


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