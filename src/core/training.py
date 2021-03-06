# I don't need that import, but i got super annoying errors in pycharm. Somehow they are gone now ¯\_(ツ)_/¯
import pylab

import sys
from datetime import datetime
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import math
from livelossplot import PlotLosses

from src.utils.config import NUM_EPOCHS, PLOT, LR_SCALE
from src.utils.logger import logger
from src.core.model import model, save_model, CrossEntropyNoSMLoss
from src.core.data import get_dataloaders
from src.optimize.compile_model import compile_model


def train(net, train_loader, test_loader, criterion, optimizer, device, start_epoch):
    start_time = datetime.now()
    plot = PlotLosses(skip_first=0)
    plot.global_step = start_epoch

    num_epochs = NUM_EPOCHS + start_epoch
    save_name = None

    min_loss, val_loss = math.inf, math.inf

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', LR_SCALE, 3)

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch: {:d}/{:d}'.format(epoch + 1, num_epochs))

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        net.train()
        loss, acc = fit(net, train_loader, criterion, optimizer, device, epoch, lr, training=True)
        net.eval()
        val_loss, val_acc = fit(net, test_loader, criterion, optimizer, device, epoch, lr, training=False)

        logger.info('val_loss: {:.3f} | val_accuracy: {:.2f}% | train_loss: {:.3f} | train_accuracy: {:.2f}%'
                    .format(val_loss, val_acc, loss, acc))

        if min_loss > val_loss and epoch > 2:
            save_name = save_model(net, epoch, loss=val_loss, name=save_name, lr=lr)
            min_loss = val_loss

        if PLOT:
            plot.update({
                'loss': loss,
                'accuracy': acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            plot.draw()
            print()

        scheduler.step(val_loss)

    if save_name is not None:
        save_model(net, num_epochs, loss=val_loss, name='final_{:s}'.format(save_name), lr=lr)
        compile_model(save_name)

    logger.info('Finished training in {}'.format((datetime.now() - start_time)))


def fit(net, data_loader, criterion, optimizer, device, epoch, lr, training=True):
    acc_sum, acc_avg, loss_sum, loss_avg = 0.0, 0.0, 0.0, math.inf

    with tqdm(data_loader, ascii=True, unit='batches', leave=True,
              bar_format='{desc:10}{percentage:3.0f}% |{bar:100}| {n_fmt:3}/{total_fmt:3} |'
                         ' time elapsed: {elapsed}, time remaining: {remaining:5}, {rate_fmt:14} |'
                         ' epoch: {postfix[0][epoch]:2d} | loss: {postfix[0][loss]:.2f} | accuracy: {postfix[0][acc]:3.2f}% '
                         '| lr: {postfix[0][lr]}',
              postfix=[dict(epoch=epoch + 1, loss=loss_avg, acc=acc_avg, lr=lr)],
              file=sys.stdout) as t:

        if training:
            t.set_description('training')
        else:
            t.set_description('test')

        for i, (images, labels) in enumerate(t):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum(dtype=torch.float32)
            accuracy = 100 * correct / len(labels)

            acc_sum += accuracy.item()
            loss_sum += loss.item()

            acc_avg = acc_sum / (i + 1)
            loss_avg = loss_sum / (i + 1)

            t.postfix[0]['loss'] = loss_avg
            t.postfix[0]['acc'] = acc_avg

        return loss_avg, acc_avg


def start_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device: {}'.format(device))

    net, optimizer, start_epoch, _ = model(device = device)

    train_loader, test_loader = get_dataloaders()

    criterion = CrossEntropyNoSMLoss()

    train(net, train_loader, test_loader, criterion, optimizer, device, start_epoch)
