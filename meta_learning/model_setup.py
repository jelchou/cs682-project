import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

def setup_resnet50(num_classes, pretrained=True, freeze_layers=True):
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
