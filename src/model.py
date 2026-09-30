import torch
import torch.nn as nn


class BreastCancerCNN(nn.Module):
    def __init__(self, variant="B"):
        super().__init__()

        self.variant = variant

        if variant == "A":
            # 2 convoluciones + 2 pooling
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.AdaptiveAvgPool2d((1, 1))
            )

            final_features = 64

        elif variant == "B":
            # 3 convoluciones + 3 pooling
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.AdaptiveAvgPool2d((1, 1))
            )

            final_features = 128

        elif variant == "C":
            # 3 convoluciones + 2 pooling
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((1, 1))
            )

            final_features = 128

        else:
            raise ValueError(
                f"Variant '{variant}' no válida. Usa A, B o C."
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(final_features, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    x = torch.randn(2, 3, 256, 256)

    for variant in ["A", "B", "C"]:
        model = BreastCancerCNN(variant=variant)
        output = model(x)

        print()
        print("===================================")
        print("Modelo:", variant)
        print("Entrada:", x.shape)
        print("Salida:", output.shape)
        print("===================================")