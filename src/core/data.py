import torch
from torchvision import transforms as T
import os
from os import path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import time

from src.utils.config import DATA_ANNOTATION, DATA_ROOT, IMAGE_SIZE, IMAGE_SCALE, BATCH_SIZE, SHUFFLE, NUM_WORKERS, TRAIN_SPLIT, RANDOM_SEED
from src.utils.enums import Person
from src.utils.logger import logger


class RaspiDataset(Dataset):

    def __init__(self, annot, rootdir=os.getcwd()):
        self.data_root = rootdir

        frame = pd.read_csv(annot)
        logger.info('loaded annotations from file: \'{:s}\''.format(annot))

        self.image_path_series = frame['image']
        self.labels = torch.tensor(frame['label'])

        self.transforms = CustomTransforms(IMAGE_SIZE, IMAGE_SCALE)

    def __getitem__(self, index):
        img_path = path.join(self.data_root, self.image_path_series[index])
        image = Image.open(img_path)
        trans_img = self.transforms(image)
        return trans_img, self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_dataloaders():
    dataset = RaspiDataset(DATA_ANNOTATION, DATA_ROOT)

    indices = list(range(len(dataset)))
    split = int(np.floor(TRAIN_SPLIT * len(dataset)))
    seed = int(time.time()) if RANDOM_SEED else 27

    if SHUFFLE:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, sampler=train_sampler, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset, sampler=test_sampler, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    logger.info('loaded dataset with {:d} train-samples and {:d} test-samples'.format(len(train_indices), len(test_indices)))

    return train_loader, test_loader

def generate_csv(absolutePath=True):
    images = []
    labels = []

    for person in Person:
        person_root = path.join(DATA_ROOT, person.name.capitalize())
        if os.path.exists(person_root):
            if absolutePath:
                images.extend([os.path.abspath(path.join(person_root, img)) for img in os.listdir(person_root)])
            else:
                images.extend([path.join(person_root, img) for img in os.listdir(person_root)])

            labels.extend([person.value for i in os.listdir(person_root)])

    data = {'image': images,
            'label': labels}

    frame = pd.DataFrame(data, columns=['image', 'label'])

    frame.to_csv(DATA_ANNOTATION)
    logger.info('Annotation created at {:s}'.format(DATA_ANNOTATION))


class CustomTransforms:
    def __init__(self, dst_size, scale):
        self.dst_size = dst_size
        self.scale = scale

    def __call__(self, image):
        ratio = float(self.dst_size * self.scale) / max(image.size)
        resize1 = (int(image.size[1] * ratio), int(image.size[0] * ratio))
        padding = (int(image.size[0] * ratio) - int(image.size[1] * ratio)) // 2

        transforms = T.Compose([
            T.Resize(resize1),
            T.Pad((0, padding, 0, padding), padding_mode='edge'),
            T.CenterCrop(self.dst_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transforms(image)
