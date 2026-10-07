import torch
import torch.nn as nn

from config.config import (
    INPUT_CHANNELS,
    CONV_CHANNELS,
    KERNEL_SIZE,
    PADDING,
    POOL_SIZE,
    USE_BATCH_NORM,
    DROPOUT,
)


class BreastCancerCNN(nn.Module):

    def __init__(self):
        super().__init__()

        layers = []
        in_channels = INPUT_CHANNELS

        # Construcción automática de los bloques convolucionales
        for out_channels in CONV_CHANNELS:

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    padding=PADDING,
                )
            )

            if USE_BATCH_NORM:
                layers.append(
                    nn.BatchNorm2d(out_channels)
                )

            layers.append(nn.ReLU())

            layers.append(
                nn.MaxPool2d(
                    kernel_size=POOL_SIZE
                )
            )

            in_channels = out_channels

        # Reduce cada mapa de características a 1x1
        layers.append(
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.features = nn.Sequential(*layers)

        # Clasificador binario
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                CONV_CHANNELS[-1],
                1
            ),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x