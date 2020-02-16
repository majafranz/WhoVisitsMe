import torch.nn as nn
from torchvision.models import resnet
from src.utils.logger import logger

def model(num_classes):
    model = resnet.resnet50(pretrained = False)
    model.fc = nn.Linear(2048, num_classes, bias=True)
    return model
