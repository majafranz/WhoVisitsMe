import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from torch.nn import _reduction
from torchvision.models import resnet
import os
from os import path
from datetime import datetime
import math

from src.utils.logger import logger
from src.utils.config import SAVE_PATH, LOAD_MODEL_PATH, SPEC_SAVE_NAME, NUM_CLASSES


def model(load_path=None):
    model = resnet.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, NUM_CLASSES, bias=True)
    model = nn.Sequential(model, nn.LogSoftmax(dim=1))

    if load_path is None and LOAD_MODEL_PATH is not None:
        load_path = LOAD_MODEL_PATH

    if load_path is not None:
        load_path = path.join(SAVE_PATH, load_path)
        ckpt = torch.load(load_path)
        model.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        logger.info('Loaded model from file: \'{:s}\''.format(load_path))

    else:
        logger.info('Training with model from scratch!')
        epoch = 0

    return model, epoch


def save_model(model, epoch=0, loss=math.inf, name=None):
    if SPEC_SAVE_NAME is not None:
        filename = SPEC_SAVE_NAME

    else:
        if name is not None:
            filename = name

        else:
            fmt = 'model_%Y%m%d_%H%M%S.pt'
            filename = datetime.now().strftime(fmt)

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    full_path = path.join(SAVE_PATH, filename)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, full_path)

    logger.info('Saved model in \'{:s}\''.format(full_path))

    return filename


class CrossEntropyNoSMLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return cross_entropy_no_sm(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def cross_entropy_no_sm(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)

    return nll_loss(input, target, weight, None, ignore_index, None, reduction)