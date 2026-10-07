from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from config.config import (
    IMAGE_SIZE,
    INPUT_MODE,
    USE_STANDARDIZATION,
    CHANNEL_MEANS,
    CHANNEL_STDS,
)


class BreastDCEDataset(Dataset):
    def __init__(
        self,
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split=None,
    ):
        """
        Dataset para BreastDCEDL.

        Modos de entrada controlados desde config.py:

        INPUT_MODE = "raw"
            canal 0 -> PRE
            canal 1 -> EARLY
            canal 2 -> LATE

        INPUT_MODE = "enhancement"
            canal 0 -> PRE
            canal 1 -> EARLY - PRE
            canal 2 -> LATE - PRE

        Si split=None se utilizan todas las filas del CSV.
        Esto es lo que usamos con internal_train.csv
        e internal_val.csv, porque ya están separados.
        """

        self.root_dir = Path(root_dir)

        self.df = pd.read_csv(
            csv_path
        )

        # =================================================
        # SPLIT
        # =================================================

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
                f"No se encontraron muestras "
                f"en {csv_path}"
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
                f"Opciones: {valid_input_modes}"
            )

        # =================================================
        # COMPROBACIÓN ESTANDARIZACIÓN
        # =================================================

        if USE_STANDARDIZATION:

            if len(CHANNEL_MEANS) != 3:

                raise ValueError(
                    "CHANNEL_MEANS debe tener 3 valores."
                )

            if len(CHANNEL_STDS) != 3:

                raise ValueError(
                    "CHANNEL_STDS debe tener 3 valores."
                )

            if any(
                std <= 0
                for std in CHANNEL_STDS
            ):

                raise ValueError(
                    "Todos los CHANNEL_STDS "
                    "deben ser mayores que 0."
                )

    def __len__(self):

        return len(self.df)

    # =====================================================
    # CARGA DE IMAGEN
    # =====================================================

    def _load_grayscale_image(
        self,
        relative_path,
    ):

        image_path = (
            self.root_dir
            / relative_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"No se encontró la imagen:\n"
                f"{image_path}"
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

        image_tensor = (
            torch.from_numpy(
                image_array
            )
        )

        # [0, 255] -> [0, 1]
        image_tensor = (
            image_tensor / 255.0
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
                f"INPUT_MODE no reconocido: "
                f"{INPUT_MODE}"
            )

        image = torch.stack(
            channels,
            dim=0,
        )

        return image

    # =====================================================
    # ESTANDARIZACIÓN
    # =====================================================

    def _standardize(
        self,
        image,
    ):

        if not USE_STANDARDIZATION:

            return image

        image = image.clone()

        for channel_index in range(3):

            mean = (
                CHANNEL_MEANS[
                    channel_index
                ]
            )

            std = (
                CHANNEL_STDS[
                    channel_index
                ]
            )

            image[
                channel_index
            ] = (
                image[
                    channel_index
                ]
                - mean
            ) / std

        return image

    # =====================================================
    # GETITEM
    # =====================================================

    def __getitem__(
        self,
        idx,
    ):

        row = (
            self.df.iloc[idx]
        )

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

        image = (
            self._build_channels(
                pre=pre,
                early=early,
                late=late,
            )
        )

        image = (
            self._standardize(
                image
            )
        )

        label = torch.tensor(
            float(
                row["pCR"]
            ),
            dtype=torch.float32,
        )

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
# TEST
# =========================================================

if __name__ == "__main__":

    dataset = BreastDCEDataset(
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split="train",
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