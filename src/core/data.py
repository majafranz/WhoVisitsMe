import torch
from torchvision import transforms as T
import os
from os import path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import time

from src.utils.config import DATA_ROOT, IMAGE_SIZE, IMAGE_SCALE, BATCH_SIZE, SHUFFLE, NUM_WORKERS, TRAIN_SPLIT, \
    RANDOM_SEED, TRANS_DATA_ROOT
from src.utils.enums import Person
from src.utils.logger import logger


class RaspiDataset(Dataset):

    def __init__(self, rootdir=os.getcwd()):
        self.data_root = rootdir

        annot = os.path.join(rootdir, 'annotation.csv')
        frame = pd.read_csv(annot)
        logger.info('loaded annotations from file: \'{:s}\''.format(annot))

        self.image_path_series = frame['image']
        self.labels = torch.tensor(frame['label'])

        self.transforms = CustomTransforms(IMAGE_SIZE, IMAGE_SCALE, transed_before=True)

    def __getitem__(self, index):
        img_path = path.join(self.data_root, self.image_path_series[index])
        image = Image.open(img_path)
        trans_img = self.transforms(image)
        return trans_img, self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_dataloaders():
    dataset = RaspiDataset(TRANS_DATA_ROOT)

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

    logger.info(
        'loaded dataset with {:d} train-samples and {:d} test-samples'.format(len(train_indices), len(test_indices)))

    return train_loader, test_loader


def generate_csv(data_root, absolutePath=True):
    images = []
    labels = []

    for person in Person:
        person_root = path.join(data_root, person.name.capitalize())
        if os.path.exists(person_root):
            if absolutePath:
                images.extend([os.path.abspath(path.join(person_root, img)) for img in os.listdir(person_root)])
            else:
                images.extend([path.join(person_root, img) for img in os.listdir(person_root)])

            labels.extend([person.value for i in os.listdir(person_root)])

    data = {'image': images,
            'label': labels}

    frame = pd.DataFrame(data, columns=['image', 'label'])

    csvfile = os.path.join(data_root, 'annotation.csv')
    frame.to_csv(csvfile)
    logger.info('Annotation created at {:s}'.format(csvfile))
    return csvfile


class CustomTransforms:
    def __init__(self, dst_size, scale, store=False, transed_before=False):
        self.dst_size = dst_size
        self.scale = scale
        self.store = store
        self.transed_before = transed_before

    def __call__(self, image):
        ratio = float(self.dst_size * self.scale) / max(image.size)
        resize1 = (int(image.size[1] * ratio), int(image.size[0] * ratio))
        padding = (int(image.size[0] * ratio) - int(image.size[1] * ratio)) // 2

        transforms = T.Compose([
            T.Resize(resize1),
            T.Pad((0, padding, 0, padding), padding_mode='edge'),
            T.CenterCrop(self.dst_size),
        ])

        if self.store:
            return transforms(image)

        else:
            if not self.transed_before:
                image = transforms(image)

            tensor_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return tensor_transforms(image)


def store_transformed_imgs():
    csvfile = generate_csv(DATA_ROOT)
    trans_csvfile = os.path.join(TRANS_DATA_ROOT, 'annotation.csv')

    if os.path.exists(csvfile):
        frame = pd.read_csv(csvfile)

        transforms = CustomTransforms(IMAGE_SIZE, IMAGE_SCALE, True)

        image_path_series = frame['image']

        if os.path.exists(trans_csvfile):
            frame = pd.read_csv(trans_csvfile)

            image_path_series_trans = frame['image']

        else:
            image_path_series_trans = None

        os.makedirs(TRANS_DATA_ROOT, exist_ok=True)

        for i, (path) in enumerate(image_path_series):
            _path = path[path.rfind('\\', 0, path.rfind('\\')) + 1:] if path.rfind('\\') > -1 \
                else path[path.rfind('/', 0, path.rfind('/')) + 1:]

            if (image_path_series_trans is not None) and \
                    (os.path.abspath(os.path.join(TRANS_DATA_ROOT, _path)) in np.array(image_path_series_trans)):
                continue

            new_path = os.path.join(TRANS_DATA_ROOT, _path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            image = Image.open(path)
            image = transforms(image)
            image.save(new_path)
            logger.info('transformed image stored in \'{:s}\''.format(new_path))

        logger.info('transformed images stored in \'{:s}\''.format(TRANS_DATA_ROOT))

        generate_csv(TRANS_DATA_ROOT)

    else:
        logger.error('OOPSI PUPSI! Seems like csv file is missing')

