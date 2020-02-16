import time
import sys
import torch
import torch.optim as optim
import torch.nn as nn

from src.utils.config import NUM_EPOCHS, NUM_CLASSES
from src.utils.logger import trlog, logger
from src.core.model import model
from src.core.data import get_dataloaders

def train(net, num_epochs, train_loader, test_loader, criterion, optimizer, device):
    start_time = time.time()
    best_acc = 0.0
    net.train()
    for epoch in range(num_epochs):
        trlog.info('Epoch: {:d}/{:d}'.format(epoch+1, num_epochs))
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            sys.stdout.write('\rBatch {:d}/{:d}'.format(i, len(train_loader)))
            sys.stdout.flush()

            optimizer.zero_grad()

            outputs = net(image)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total = len(label)
            correct = (predicted == label).sum(dtype=torch.float32)
            accuracy = 100 * correct / total

            if not i % 25:
                trlog.info('epoch {:d}, step {:d}: loss={:.2f}, accuracy={:.2f}'.format(epoch+1, i, loss.item(), accuracy.item()))



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