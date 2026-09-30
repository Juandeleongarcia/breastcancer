from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

CSV_PATH = "metadata/samples.csv"
OUTPUT_DIR = Path("metadata")

VAL_SIZE = 0.20
RANDOM_SEED = 42


def main():

    # ---------------------------------------------------------
    # 1. Cargar metadata
    # ---------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    # Usamos EXCLUSIVAMENTE el train original
    train_original = df[df["split"] == "train"].copy()

    print("Muestras del train original:", len(train_original))
    print(
        "Pacientes del train original:",
        train_original["patient_id"].nunique()
    )

    # ---------------------------------------------------------
    # 2. Comprobar que cada paciente tiene una única etiqueta
    # ---------------------------------------------------------

    labels_per_patient = (
        train_original
        .groupby("patient_id")["pCR"]
        .nunique()
    )

    problematic_patients = labels_per_patient[
        labels_per_patient > 1
    ]

    if len(problematic_patients) > 0:
        raise ValueError(
            "Hay pacientes con más de una etiqueta pCR. "
            "Revisar metadata antes de continuar."
        )

    # ---------------------------------------------------------
    # 3. Crear tabla a nivel PACIENTE
    # ---------------------------------------------------------

    patients = (
        train_original[
            ["patient_id", "pCR"]
        ]
        .drop_duplicates(subset="patient_id")
        .reset_index(drop=True)
    )

    print()
    print("Distribución por paciente:")
    print(patients["pCR"].value_counts().sort_index())

    # ---------------------------------------------------------
    # 4. Split 80/20 a nivel paciente
    #
    # stratify mantiene aproximadamente la proporción
    # pCR / no pCR en ambas particiones.
    # ---------------------------------------------------------

    train_patients, val_patients = train_test_split(
        patients,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=patients["pCR"]
    )

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])

    # ---------------------------------------------------------
    # 5. Comprobar que NO hay pacientes repetidos
    # ---------------------------------------------------------

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise RuntimeError(
            "ERROR: existen pacientes presentes tanto "
            "en train como en validación."
        )

    # ---------------------------------------------------------
    # 6. Recuperar TODOS los cortes de cada paciente
    # ---------------------------------------------------------

    internal_train = train_original[
        train_original["patient_id"].isin(train_ids)
    ].copy()

    internal_val = train_original[
        train_original["patient_id"].isin(val_ids)
    ].copy()

    # Indicamos claramente la nueva partición
    internal_train["internal_split"] = "train"
    internal_val["internal_split"] = "validation"

    # ---------------------------------------------------------
    # 7. Guardar CSV independientes
    # ---------------------------------------------------------

    train_output = OUTPUT_DIR / "internal_train.csv"
    val_output = OUTPUT_DIR / "internal_val.csv"

    internal_train.to_csv(
        train_output,
        index=False
    )

    internal_val.to_csv(
        val_output,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Resumen
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("        SPLIT INTERNO COMPLETADO")
    print("========================================")

    print()
    print("TRAIN INTERNO")
    print(
        "Pacientes:",
        internal_train["patient_id"].nunique()
    )
    print(
        "Muestras:",
        len(internal_train)
    )

    print()
    print(
        "Pacientes por clase:"
    )

    print(
        internal_train[
            ["patient_id", "pCR"]
        ]
        .drop_duplicates()
        ["pCR"]
        .value_counts()
        .sort_index()
    )

    print()
    print("VALIDACIÓN INTERNA")

    print(
        "Pacientes:",
        internal_val["patient_id"].nunique()
    )

    print(
        "Muestras:",
        len(internal_val)
    )

    print()
    print(
        "Pacientes por clase:"
    )

    print(
        internal_val[
            ["patient_id", "pCR"]
        ]
        .drop_duplicates()
        ["pCR"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Pacientes compartidos entre train y validation:",
        len(overlap)
    )

    # ---------------------------------------------------------
    # 9. Calcular pos_weight
    # ---------------------------------------------------------

    patient_labels_train = (
        internal_train[
            ["patient_id", "pCR"]
        ]
        .drop_duplicates()
    )

    n0 = (
        patient_labels_train["pCR"] == 0
    ).sum()

    n1 = (
        patient_labels_train["pCR"] == 1
    ).sum()

    pos_weight = n0 / n1

    print()
    print("N0 (no pCR):", n0)
    print("N1 (pCR):", n1)
    print("pos_weight N0/N1:", pos_weight)

    print()
    print("Archivos creados:")
    print(train_output)
    print(val_output)


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# NOTA SOBRE LA PARTICIÓN
#
# La unidad estadística del problema es la PACIENTE, no cada corte.
#
# Cada paciente puede tener varios cortes de MRI. Por ello no se puede
# dividir aleatoriamente por imágenes, ya que algunos cortes de una misma
# paciente podrían terminar en entrenamiento y otros en validación.
#
# Eso produciría data leakage.
#
# Este script:
#
#   1. utiliza únicamente el split "train" original;
#   2. identifica pacientes únicos;
#   3. realiza una división 80 % / 20 % a nivel paciente;
#   4. mantiene aproximadamente la proporción pCR / no pCR mediante
#      estratificación;
#   5. devuelve después todos los cortes de cada paciente a su partición;
#   6. comprueba que no existen pacientes compartidos.
#
# El conjunto TEST original NO se utiliza para elegir arquitectura,
# hiperparámetros, época de parada ni umbral.
#
# RANDOM_SEED = 42 permite reproducir exactamente la misma partición.
# -------------------------------------------------------------------------