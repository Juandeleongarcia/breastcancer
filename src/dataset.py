from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BreastDCEDataset(Dataset):
    def __init__(
        self,
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split="train"
    ):
        """
        Dataset para cargar muestras BreastDCEDL.

        Cada muestra está formada por tres fases de DCE-MRI:
        PRE, EARLY y LATE.

        Las tres imágenes se apilan como tres canales para formar
        un tensor de tamaño [3, 256, 256].
        """

        self.root_dir = Path(root_dir)

        # Leemos el CSV con la información de todas las muestras
        self.df = pd.read_csv(csv_path)

        # Filtramos por el split deseado
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No se encontraron muestras para el split '{split}'"
            )

    def __len__(self):
        """
        Devuelve el número total de muestras disponibles.
        """
        return len(self.df)

    def _load_grayscale_image(self, relative_path):
        """
        Carga una imagen PNG en escala de grises.

        La imagen original está almacenada en uint8 con valores
        entre 0 y 255.

        Se convierte a float32 y se normaliza al intervalo [0, 1].
        """

        image_path = self.root_dir / relative_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"No se encontró la imagen:\n{image_path}"
            )

        # Abrimos la imagen y forzamos escala de grises
        image = Image.open(image_path).convert("L")

        # Comprobamos el tamaño esperado
        if image.size != (256, 256):
            raise ValueError(
                f"La imagen {image_path} tiene tamaño {image.size}, "
                "pero se esperaba (256, 256)"
            )

        # Convertimos la imagen PIL a array NumPy float32
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Convertimos a tensor PyTorch
        image_tensor = torch.from_numpy(image_array)

        # Normalización [0, 255] -> [0, 1]
        image_tensor = image_tensor / 255.0

        return image_tensor

    def __getitem__(self, idx):
        """
        Devuelve una muestra del dataset.
        """

        row = self.df.iloc[idx]

        # Cargamos las tres fases correspondientes
        # al mismo corte anatómico
        pre = self._load_grayscale_image(row["path_pre"])
        early = self._load_grayscale_image(row["path_early"])
        late = self._load_grayscale_image(row["path_late"])

        # Apilamos las tres fases en el orden correcto:
        # canal 0 -> PRE
        # canal 1 -> EARLY
        # canal 2 -> LATE
        #
        # Resultado: [3, 256, 256]
        image = torch.stack(
            [pre, early, late],
            dim=0
        )

        # Etiqueta binaria:
        # 1 -> pCR
        # 0 -> no pCR
        label = torch.tensor(
            float(row["pCR"]),
            dtype=torch.float32
        )

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"],
            "sample_id": row["sample_id"],
            "slice_index": row["slice_index"]
        }


if __name__ == "__main__":

    # Creamos el dataset de entrenamiento
    dataset = BreastDCEDataset(
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split="train"
    )

    print("Número de muestras:", len(dataset))

    # Cargamos la primera muestra real
    sample = dataset[0]

    print("Sample ID:", sample["sample_id"])
    print("Patient ID:", sample["patient_id"])
    print("Slice:", sample["slice_index"])

    print("Forma de la imagen:", sample["image"].shape)
    print("Etiqueta pCR:", sample["label"])

    print("Valor mínimo:", sample["image"].min().item())
    print("Valor máximo:", sample["image"].max().item())

    print(
        "Media PRE:",
        sample["image"][0].mean().item()
    )

    print(
        "Media EARLY:",
        sample["image"][1].mean().item()
    )

    print(
        "Media LATE:",
        sample["image"][2].mean().item()
    )


# -------------------------------------------------------------------------
# NOTA SOBRE EL DATASET
#
# Cada muestra está formada por tres imágenes correspondientes al mismo
# corte anatómico de una paciente:
#
#   canal 0 -> PRE
#   canal 1 -> EARLY
#   canal 2 -> LATE
#
# Las imágenes son PNG en escala de grises de tamaño 256x256.
#
# Los valores originales se encuentran entre 0 y 255 y se dividen entre
# 255 para trabajar en el intervalo [0, 1].
#
# Las tres fases se apilan para obtener un tensor:
#
#   [3, 256, 256]
#
# Este es el formato de entrada que utiliza nuestra CNN.
#
# Es fundamental mantener PRE, EARLY y LATE correctamente alineadas.
# Si posteriormente se aplica data augmentation, cualquier transformación
# geométrica deberá aplicarse exactamente igual a las tres fases.
#
# La etiqueta pCR es binaria:
#
#   pCR = 1  -> respuesta patológica completa
#   pCR = 0  -> no respuesta patológica completa
#
# La división train/validation/test debe respetar siempre la unidad paciente
# para evitar data leakage entre cortes de una misma paciente.
# -------------------------------------------------------------------------