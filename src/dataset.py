from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from config.config import (
    IMAGE_SIZE,
    INPUT_MODE,
    USE_CLIPPING,
    CHANNEL_P01,
    CHANNEL_P99,
    USE_STANDARDIZATION,
    CHANNEL_MEANS,
    CHANNEL_STDS,
)


class BreastDCEDataset(Dataset):
    """
    Dataset definitivo para BreastDCEDL.

    Pipeline:

    1. Carga PRE / EARLY / LATE.
    2. Convierte intensidades de [0, 255] a [0, 1].
    3. Construye los canales según INPUT_MODE.
    4. Aplica clipping opcional por canal.
    5. Aplica estandarización opcional por canal.

    Todo se controla desde config/config.py.
    """

    def __init__(
        self,
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split=None,
    ):

        self.root_dir = Path(root_dir)

        self.df = pd.read_csv(csv_path)

        # =================================================
        # SPLIT
        # =================================================

        # Si utilizamos internal_train.csv o internal_val.csv,
        # pasaremos split=None porque ya están separados.
        #
        # Si usamos samples.csv y contiene columna "split",
        # sí podemos filtrar por train/val/test.

        if (
            split is not None
            and "split" in self.df.columns
        ):

            self.df = (
                self.df[
                    self.df["split"] == split
                ]
                .reset_index(drop=True)
            )

        else:

            self.df = (
                self.df
                .reset_index(drop=True)
            )

        if len(self.df) == 0:

            raise ValueError(
                f"No se encontraron muestras en {csv_path}"
            )

        # =================================================
        # COMPROBACIONES DE COLUMNAS
        # =================================================

        required_columns = [
            "path_pre",
            "path_early",
            "path_late",
            "pCR",
            "patient_id",
            "sample_id",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Faltan columnas obligatorias en el CSV: "
                f"{missing_columns}"
            )

        # =================================================
        # COMPROBACIÓN INPUT MODE
        # =================================================

        valid_input_modes = [
            "raw",
            "enhancement",
        ]

        if INPUT_MODE not in valid_input_modes:

            raise ValueError(
                f"INPUT_MODE='{INPUT_MODE}' no válido. "
                f"Opciones disponibles: {valid_input_modes}"
            )

        # =================================================
        # COMPROBACIÓN CLIPPING
        # =================================================

        if USE_CLIPPING:

            if len(CHANNEL_P01) != 3:

                raise ValueError(
                    "CHANNEL_P01 debe contener 3 valores."
                )

            if len(CHANNEL_P99) != 3:

                raise ValueError(
                    "CHANNEL_P99 debe contener 3 valores."
                )

            for channel_index in range(3):

                if (
                    CHANNEL_P01[channel_index]
                    >= CHANNEL_P99[channel_index]
                ):

                    raise ValueError(
                        f"Canal {channel_index}: "
                        "CHANNEL_P01 debe ser menor "
                        "que CHANNEL_P99."
                    )

        # =================================================
        # COMPROBACIÓN ESTANDARIZACIÓN
        # =================================================

        if USE_STANDARDIZATION:

            if len(CHANNEL_MEANS) != 3:

                raise ValueError(
                    "CHANNEL_MEANS debe contener 3 valores."
                )

            if len(CHANNEL_STDS) != 3:

                raise ValueError(
                    "CHANNEL_STDS debe contener 3 valores."
                )

            if any(
                std <= 0
                for std in CHANNEL_STDS
            ):

                raise ValueError(
                    "Todos los CHANNEL_STDS deben ser > 0."
                )

    def __len__(self):

        return len(self.df)

    # =====================================================
    # CARGA DE IMÁGENES
    # =====================================================

    def _load_grayscale_image(
        self,
        relative_path,
    ):
        """
        Carga una imagen PNG en escala de grises.

        Entrada:
            uint8 [0, 255]

        Salida:
            float32 [0, 1]
        """

        image_path = (
            self.root_dir
            / relative_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"No se encontró la imagen:\n{image_path}"
            )

        image = (
            Image.open(image_path)
            .convert("L")
        )

        expected_size = (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        if image.size != expected_size:

            raise ValueError(
                f"La imagen {image_path} "
                f"tiene tamaño {image.size}, "
                f"pero se esperaba {expected_size}"
            )

        image_array = np.array(
            image,
            dtype=np.float32,
        )

        image_tensor = torch.from_numpy(
            image_array
        )

        # Normalización básica:
        # [0, 255] -> [0, 1]

        image_tensor = (
            image_tensor
            / 255.0
        )

        return image_tensor

    # =====================================================
    # CONSTRUCCIÓN DE CANALES
    # =====================================================

    def _build_channels(
        self,
        pre,
        early,
        late,
    ):
        """
        Construye los tres canales que recibe la CNN.

        INPUT_MODE = "raw"

            canal 0 -> PRE
            canal 1 -> EARLY
            canal 2 -> LATE

        INPUT_MODE = "enhancement"

            canal 0 -> PRE
            canal 1 -> EARLY - PRE
            canal 2 -> LATE - PRE
        """

        if INPUT_MODE == "raw":

            channels = [
                pre,
                early,
                late,
            ]

        elif INPUT_MODE == "enhancement":

            early_enhancement = (
                early - pre
            )

            late_enhancement = (
                late - pre
            )

            channels = [
                pre,
                early_enhancement,
                late_enhancement,
            ]

        else:

            raise ValueError(
                f"INPUT_MODE no reconocido: {INPUT_MODE}"
            )

        image = torch.stack(
            channels,
            dim=0,
        )

        return image

    # =====================================================
    # CLIPPING
    # =====================================================

    def _clip_channels(
        self,
        image,
    ):
        """
        Recorta los valores extremos de cada canal.

        Los límites P01 y P99 deben haberse calculado
        utilizando únicamente TRAIN.
        """

        if not USE_CLIPPING:

            return image

        image = image.clone()

        for channel_index in range(3):

            image[channel_index] = torch.clamp(
                image[channel_index],
                min=CHANNEL_P01[channel_index],
                max=CHANNEL_P99[channel_index],
            )

        return image

    # =====================================================
    # ESTANDARIZACIÓN
    # =====================================================

    def _standardize(
        self,
        image,
    ):
        """
        Estandarización independiente por canal:

            (x - mean) / std

        mean y std deben calcularse exclusivamente
        con el conjunto de entrenamiento.
        """

        if not USE_STANDARDIZATION:

            return image

        image = image.clone()

        for channel_index in range(3):

            mean = CHANNEL_MEANS[
                channel_index
            ]

            std = CHANNEL_STDS[
                channel_index
            ]

            image[channel_index] = (
                image[channel_index]
                - mean
            ) / std

        return image

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]

        # -------------------------------------------------
        # CARGA DE PRE / EARLY / LATE
        # -------------------------------------------------

        pre = (
            self._load_grayscale_image(
                row["path_pre"]
            )
        )

        early = (
            self._load_grayscale_image(
                row["path_early"]
            )
        )

        late = (
            self._load_grayscale_image(
                row["path_late"]
            )
        )

        # -------------------------------------------------
        # CONSTRUCCIÓN DE LOS CANALES
        # -------------------------------------------------

        image = (
            self._build_channels(
                pre=pre,
                early=early,
                late=late,
            )
        )

        # -------------------------------------------------
        # CLIPPING
        # -------------------------------------------------

        image = (
            self._clip_channels(
                image
            )
        )

        # -------------------------------------------------
        # ESTANDARIZACIÓN
        # -------------------------------------------------

        image = (
            self._standardize(
                image
            )
        )

        # -------------------------------------------------
        # LABEL
        # -------------------------------------------------

        label = torch.tensor(
            float(
                row["pCR"]
            ),
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # SAMPLE
        # -------------------------------------------------

        sample = {
            "image":
                image,

            "label":
                label,

            "patient_id":
                row["patient_id"],

            "sample_id":
                row["sample_id"],
        }

        if "slice_index" in row.index:

            sample[
                "slice_index"
            ] = (
                row["slice_index"]
            )

        return sample


# =========================================================
# TEST DEL DATASET
# =========================================================

if __name__ == "__main__":

    dataset = BreastDCEDataset(
        csv_path="metadata/internal_train.csv",
        root_dir="breastdcedl",
        split=None,
    )

    print()

    print(
        "========================================"
    )

    print(
        "        TEST BREASTDCEDATASET"
    )

    print(
        "========================================"
    )

    print()

    print(
        "INPUT_MODE:",
        INPUT_MODE,
    )

    print(
        "USE_CLIPPING:",
        USE_CLIPPING,
    )

    print(
        "USE_STANDARDIZATION:",
        USE_STANDARDIZATION,
    )

    print(
        "Número de muestras:",
        len(dataset),
    )

    print()

    sample = dataset[0]

    print(
        "Sample ID:",
        sample["sample_id"],
    )

    print(
        "Patient ID:",
        sample["patient_id"],
    )

    if "slice_index" in sample:

        print(
            "Slice:",
            sample["slice_index"],
        )

    print(
        "Forma:",
        sample["image"].shape,
    )

    print(
        "Etiqueta:",
        sample["label"],
    )

    print()

    # =====================================================
    # ESTADÍSTICAS DE LA PRIMERA MUESTRA
    # =====================================================

    for channel_index in range(3):

        channel = (
            sample["image"][
                channel_index
            ]
        )

        print(
            f"Canal {channel_index}"
        )

        print(
            "  Min:",
            channel.min().item(),
        )

        print(
            "  Max:",
            channel.max().item(),
        )

        print(
            "  Mean:",
            channel.mean().item(),
        )

        print(
            "  Std:",
            channel.std().item(),
        )

        print()