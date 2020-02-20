import torch
import torch.nn as nn
from torchvision.models import resnet
import os
from os import path
from datetime import datetime
import math

from src.utils.logger import logger
from src.utils.config import SAVE_PATH, LOAD_MODEL_PATH, SPEC_SAVE_NAME

def model(num_classes):
    model = resnet.resnet50(pretrained = False)
    model.fc = nn.Linear(2048, num_classes, bias=True)

    if LOAD_MODEL_PATH is not None:
        load_path= path.join(SAVE_PATH, LOAD_MODEL_PATH)
        ckpt = torch.load(load_path)
        model.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        logger.info('Loaded model from file: {:s}'.format(load_path))

    else:
        epoch = 0

    return model, epoch


def save_model(model, epoch=0, loss=math.inf, acc=0.0):

    if SPEC_SAVE_NAME is None:
        fmt = 'model_%Y%m%d_%H%M%S.pt'
        filename = datetime.now().strftime(fmt)

    else:
        filename= SPEC_SAVE_NAME

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    full_path = path.join(SAVE_PATH, filename)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, full_path)

    logger.info('Saved model in {:s}'.format(full_path))