from typing import Tuple
import torch.nn as nn
from torchvision import models

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, str]:
    name = model_name.lower()
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        target_layer = "layer4.1.conv2"
    elif name == "mobilenetv3":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)
        target_layer = "features.12"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return net, target_layer
