import time
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math

from src.utils.config import NUM_EPOCHS, NUM_CLASSES
from src.utils.logger import trlog, logger
from src.core.model import model
from src.core.data import get_dataloaders


def train(net, num_epochs, train_loader, test_loader, criterion, optimizer, device):
    start_time = time.time()

    for epoch in range(num_epochs):
        trlog.info('Epoch: {:d}/{:d}'.format(epoch + 1, num_epochs))
        net.train()
        fit(net, train_loader, criterion, optimizer, device, epoch, training=True)
        net.eval()
        fit(net, test_loader, criterion, optimizer, device, epoch, training=False)


def fit(net, data_loader, criterion, optimizer, device, epoch, training=True):

    acc_sum, acc_avg, loss_sum, loss_avg = 0.0, 0.0, 0.0, math.inf

    with tqdm(data_loader, ascii=True, unit='batches',
              bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}| {n_fmt:3}/{total_fmt:3} |'
                         ' time elapsed: {elapsed}, time remaining: {remaining:5}, {rate_fmt:14} |'
                         ' epoch: {postfix[0][epoch]:2d} | loss: {postfix[0][loss]:.2f} | accuracy: {postfix[0][acc]:3.2f}',
              postfix=[dict(epoch=epoch+1, loss=loss_avg, acc=acc_avg)]) as t:

        if training:
            t.set_description('training')
        else:
            t.set_description('training')

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


def start_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device: {}'.format(device))

    net = model(NUM_CLASSES)
    net.to(device)

    train_loader, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, NUM_EPOCHS, train_loader, test_loader, criterion, optimizer, device)


if __name__ == '__main__':
    start_training()
