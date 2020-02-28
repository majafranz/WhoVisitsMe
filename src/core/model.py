import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from torch.nn import _reduction
import torch.optim as optim
from torchvision.models import resnet
import os
from os import path
from datetime import datetime
import math

from src.utils.logger import logger
from src.utils.config import SAVE_PATH, LOAD_MODEL_PATH, SPEC_SAVE_NAME, NUM_CLASSES, LR, MOMENTUM


def model(device, load_path=None):
    model = resnet.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, NUM_CLASSES, bias=True)
    model = nn.Sequential(model, nn.Softmax(dim=1))

    if load_path is None and LOAD_MODEL_PATH is not None:
        load_path = LOAD_MODEL_PATH

    if load_path is not None:
        load_path = path.join(SAVE_PATH, load_path)
        if device is torch.device('cuda'):
            ckpt = torch.load(load_path)

        else:
            ckpt = torch.load(load_path, map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        val_loss = ckpt['loss']
        lr = ckpt['lr']
        logger.info('Loaded model from file: \'{:s}\' in epoch: {:d} with loss: {:.2f}'.format(load_path, epoch, val_loss))

    else:
        logger.info('Training with model from scratch!')
        epoch = 0
        val_loss = math.inf
        lr = LR

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    return model, optimizer, epoch, val_loss


def save_model(model, epoch=0, loss=math.inf, name=None, lr=LR):
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
                'loss': loss,
                'lr': lr,
                }, full_path)

    logger.info('Saved model in \'{:s}\''.format(full_path))

    return filename


class CrossEntropyNoSMLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return cross_entropy_no_sm(torch.log(input), target, weight=self.weight,
                                   ignore_index=self.ignore_index, reduction=self.reduction)


def cross_entropy_no_sm(input, target, weight=None, size_average=None, ignore_index=-100,
                        reduce=None, reduction='mean'):
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)

    return nll_loss(input, target, weight, None, ignore_index, None, reduction)
