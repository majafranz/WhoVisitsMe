import torch.nn as nn
from torchvision.models import resnet


def model(num_classes):
    model = resnet.resnet18(pretrained = False)
    model.fc = nn.Linear(512, num_classes, bias=True)
    return model