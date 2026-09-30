import torch

from dataset import BreastDCEDataset
from model import BreastCancerCNN


def main():
    # ---------------------------------------------------------
    # 1. Cargamos el dataset real
    # ---------------------------------------------------------
    dataset = BreastDCEDataset(
        csv_path="metadata/samples.csv",
        root_dir="breastdcedl",
        split="train"
    )

    # Tomamos una muestra real
    sample = dataset[0]

    image = sample["image"]
    label = sample["label"]

    print("Sample ID:", sample["sample_id"])
    print("Patient ID:", sample["patient_id"])
    print("Forma original:", image.shape)
    print("Etiqueta real pCR:", label.item())

    # ---------------------------------------------------------
    # 2. Añadimos dimensión de batch
    #
    # La CNN espera:
    # [batch_size, canales, alto, ancho]
    #
    # Antes:
    # [3, 256, 256]
    #
    # Después:
    # [1, 3, 256, 256]
    # ---------------------------------------------------------
    image = image.unsqueeze(0)

    print("Forma para la CNN:", image.shape)

    # ---------------------------------------------------------
    # 3. Creamos la CNN
    # ---------------------------------------------------------
    model = BreastCancerCNN()

    # Modo evaluación
    model.eval()

    # ---------------------------------------------------------
    # 4. Pasamos la muestra real por la red
    # ---------------------------------------------------------
    with torch.no_grad():
        logit = model(image)

        # Convertimos el logit a probabilidad
        probability = torch.sigmoid(logit)

    print("Logit:", logit.item())
    print("Probabilidad pCR:", probability.item())

    # ---------------------------------------------------------
    # 5. Clase usando umbral 0.5
    # ---------------------------------------------------------
    threshold = 0.5

    predicted_class = int(
        probability.item() >= threshold
    )

    print("Umbral:", threshold)
    print("Clase predicha:", predicted_class)

    print()
    print("Interpretación:")
    print("0 -> no pCR")
    print("1 -> pCR")


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# NOTA SOBRE ESTE TEST
#
# Este archivo comprueba que el pipeline completo funciona correctamente:
#
#   imágenes PRE, EARLY y LATE
#               ↓
#          dataset.py
#               ↓
#       tensor [3, 256, 256]
#               ↓
#      dimensión de batch
#               ↓
#      [1, 3, 256, 256]
#               ↓
#           model.py
#               ↓
#          un único logit
#               ↓
#            sigmoid
#               ↓
#      probabilidad de pCR
#
# IMPORTANTE:
#
# La red todavía NO está entrenada.
#
# Por tanto, la probabilidad que aparezca en esta prueba NO tiene ningún
# significado predictivo. El objetivo actual es únicamente comprobar que
# una muestra real del dataset puede atravesar correctamente todo el
# pipeline sin errores.
# -------------------------------------------------------------------------