"""CNN DCE-MRI experimental: cuatro bloques, sin pesos preentrenados."""
from torch import nn

ARCHITECTURE = "F11_CNN4"
BLOCK_CHANNELS = [[8], [8, 16], [16, 32], [32, 64, 64]]
POOL_AFTER_BLOCK = [True, True, True, False]
TRAINABLE_PARAMETERS = 73913


class DCEFourBlockCNN(nn.Module):
    """Tres canales temporales; salida [batch, 1] de logits sin Sigmoid."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        if cfg.get("INPUT_CHANNELS", 3) != 3 or cfg.get("DROPOUT", .30) != .30:
            raise ValueError("F11_CNN4 requiere tres canales DCE y dropout 0.30")
        blocks = []
        incoming = 3
        for channels, pool in zip(BLOCK_CHANNELS, POOL_AFTER_BLOCK):
            layers = []
            for outgoing in channels:
                layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                               nn.BatchNorm2d(outgoing), nn.ReLU()])
                incoming = outgoing
            if pool:
                layers.append(nn.MaxPool2d(2))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(.30), nn.Linear(64, 1))

    def forward(self, images):
        return self.classifier(self.blocks(images))
