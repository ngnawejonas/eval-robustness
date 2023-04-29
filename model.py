import torch.nn as nn
from torchvision import transforms


class NormalizationLayer(nn.Module):
    """ Custom  layer """
    def __init__(self, mean=None, std=None):
        super().__init__()
        if mean is None or std is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
        
        self.transfrom = transforms.Normalize(mean = mean, std = std)

    def forward(self, x):
        return self.transfrom(x)

class ResNetXNormed(nn.Module):
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        self.model=model
        self.normalization_layer = NormalizationLayer(mean, std)

    def forward(self, x):
        x = self.normalization_layer(x)
        logits = self.model(x)
        return logits