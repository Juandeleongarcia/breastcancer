import numpy as np

from src.dataset import BreastDCEDataset
from config.config import (
    TRAIN_CSV,
    DATA_DIR,
)


def main():

    print()
    print("========================================")
    print(" ESTADÍSTICAS POST-CLIPPING SOBRE TRAIN")
    print("========================================")
    print()

    dataset = BreastDCEDataset(
        csv_path=TRAIN_CSV,
        root_dir=DATA_DIR,
        split=None,
    )

    print(
        "Muestras:",
        len(dataset),
    )

    print()

    # -----------------------------------------------------
    # Acumuladores
    # -----------------------------------------------------

    channel_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_squared_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    channel_pixel_count = np.zeros(
        3,
        dtype=np.int64,
    )

    channel_min = np.full(
        3,
        np.inf,
        dtype=np.float64,
    )

    channel_max = np.full(
        3,
        -np.inf,
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # RECORRER TODO TRAIN
    # -----------------------------------------------------

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        image = (
            sample["image"]
            .numpy()
        )

        for channel_index in range(3):

            values = (
                image[channel_index]
                .astype(
                    np.float64
                )
            )

            channel_sum[
                channel_index
            ] += values.sum()

            channel_squared_sum[
                channel_index
            ] += np.square(
                values
            ).sum()

            channel_pixel_count[
                channel_index
            ] += values.size

            channel_min[
                channel_index
            ] = min(
                channel_min[
                    channel_index
                ],
                values.min(),
            )

            channel_max[
                channel_index
            ] = max(
                channel_max[
                    channel_index
                ],
                values.max(),
            )

        if (
            (index + 1) % 500 == 0
            or
            (index + 1) == len(dataset)
        ):

            print(
                f"Procesadas "
                f"{index + 1}/"
                f"{len(dataset)} muestras"
            )

    # -----------------------------------------------------
    # MEAN Y STD
    # -----------------------------------------------------

    means = (
        channel_sum
        / channel_pixel_count
    )

    variances = (
        channel_squared_sum
        / channel_pixel_count
    ) - np.square(
        means
    )

    # Por pequeños errores numéricos
    # evitamos varianzas negativas minúsculas.
    variances = np.maximum(
        variances,
        0.0,
    )

    stds = np.sqrt(
        variances
    )

    # -----------------------------------------------------
    # RESULTADOS
    # -----------------------------------------------------

    print()
    print("========================================")
    print("        RESULTADOS DEFINITIVOS")
    print("========================================")
    print()

    for channel_index in range(3):

        print(
            f"CANAL {channel_index}"
        )

        print(
            f"Mean: "
            f"{means[channel_index]:.10f}"
        )

        print(
            f"Std:  "
            f"{stds[channel_index]:.10f}"
        )

        print(
            f"Min:  "
            f"{channel_min[channel_index]:.10f}"
        )

        print(
            f"Max:  "
            f"{channel_max[channel_index]:.10f}"
        )

        print()

    print(
        "CHANNEL_MEANS = ["
    )

    for value in means:

        print(
            f"    {value:.10f},"
        )

    print(
        "]"
    )

    print()

    print(
        "CHANNEL_STDS = ["
    )

    for value in stds:

        print(
            f"    {value:.10f},"
        )

    print(
        "]"
    )


if __name__ == "__main__":

    main()