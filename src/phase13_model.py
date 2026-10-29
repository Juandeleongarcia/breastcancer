"""SPATIAL_SMALL desde cero, extractor intacto y fusión lineal multimodal."""
import torch
from torch import nn

from src.model import BreastCancerCNN

ARCHITECTURE = 'SPATIAL_SMALL_MULTIMODAL'


class SpatialSmallMultimodal(nn.Module):
    def __init__(self, config, clinical_dim):
        super().__init__()
        required = dict(CONV_CHANNELS=[16,32,64], CONVS_PER_BLOCK=2, INPUT_CHANNELS=3,
                        SPATIAL_GRID=1, USE_BATCH_NORM=True, KERNEL_SIZE=3, PADDING=1, POOL_SIZE=2)
        if any(config.get(k) != v for k,v in required.items()) or clinical_dim < 1:
            raise ValueError('El extractor debe ser exactamente SPATIAL_SMALL')
        original = BreastCancerCNN(config)
        self.features = original.features
        self.image_dim = 64
        self.clinical_dim = int(clinical_dim)
        self.classifier = nn.Sequential(nn.Dropout(config['DROPOUT']),nn.Linear(64+clinical_dim,1))

    def forward(self, images, clinical):
        if (images.ndim != 4 or images.shape[1] != 3 or clinical.ndim != 2
                or clinical.shape != (images.shape[0],self.clinical_dim)):
            raise ValueError('Entradas esperadas [B,3,H,W] y [B,D_clínico], mismo paciente por fila')
        image_features = self.features(images).flatten(1)
        return self.classifier(torch.cat([image_features,clinical],dim=1))


def architecture_report(config, clinical_dim):
    model = SpatialSmallMultimodal(config, clinical_dim)
    original = BreastCancerCNN(config)
    return dict(architecture=ARCHITECTURE, original_parameters=sum(p.numel() for p in original.parameters()),
        parameters=sum(p.numel() for p in model.parameters()), image_features=64,
        clinical_features=clinical_dim, fused_features=64+clinical_dim, output='[B,1] logit',
        convolutions=[dict(in_channels=m.in_channels,out_channels=m.out_channels,kernel=list(m.kernel_size),
                           bias=m.bias is not None) for m in model.features if isinstance(m,nn.Conv2d)],
        batchnorm_layers=sum(isinstance(m,nn.BatchNorm2d) for m in model.features),
        maxpool_layers=3, adaptive_pool=[1,1], dropout=config['DROPOUT'], pretrained=False)
