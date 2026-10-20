import torch
import torch.nn as nn

from src.configuration import resolve_config




class BreastCancerCNN(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        cfg = resolve_config(config)
        self.config = cfg
        layers = []
        in_channels = cfg["INPUT_CHANNELS"]

        # Construcción automática de los bloques convolucionales
        for out_channels in cfg["CONV_CHANNELS"]:
            # Una convolución conserva las claves y operaciones del baseline histórico.
            # Dos amplían el campo receptivo antes de cada reducción espacial.
            for _ in range(cfg["CONVS_PER_BLOCK"]):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=cfg["KERNEL_SIZE"], padding=cfg["PADDING"]))
                if cfg["USE_BATCH_NORM"]:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                in_channels = out_channels

            layers.append(
                nn.MaxPool2d(
                    kernel_size=cfg["POOL_SIZE"]
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
            nn.Dropout(cfg["DROPOUT"]),
            nn.Linear(
                cfg["CONV_CHANNELS"][-1],
                1
            ),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x
