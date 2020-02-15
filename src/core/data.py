import torch
from torchvision import transforms as T
import os
from os import sys, path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from src.utils.config import DATA_ANNOTATION, DATA_ROOT, TRANSFORMS
from src.utils.enums import Person
from src.utils.logger import logger

class RaspiDataset(Dataset):

    def __init__(self, annot, rootdir=os.getcwd(), transforms=None):

        self.data_root = rootdir
        self.transforms = transforms

        frame = pd.read_csv(annot)
        logger.info('loaded annotations from file: \'{:s}\''.format(annot))

        self.image_path_series = frame['image']
        self.labels = torch.tensor(frame['label'])

        if transforms == None:
            self.transforms = T.Compose([
                T.ToTensor(),
            ])

    def __getitem__(self, index):
        img_path = path.join(self.data_root, self.image_path_series[index])
        image = Image.open(img_path)
        trans_img = self.transforms(image)
        return trans_img, self.labels[index]

    def __len__(self):
        return len(self.labels)


def dataset():
    dataset= RaspiDataset(DATA_ANNOTATION, DATA_ROOT, TRANSFORMS)

def generate_csv(absolutePath = True):
    images = []
    labels = []
    print(os.getcwd())
    for person in Person:
        person_root = path.join(DATA_ROOT, person.name.capitalize())
        if(os.path.exists(person_root)):
            if absolutePath:
                images.extend([path.join(os.getcwd(), person_root, img) for img in os.listdir(person_root)])
            else:
                images.extend([path.join(person_root, img) for img in os.listdir(person_root)])

            labels.extend([person.value for i in os.listdir(person_root)])

    data = {'image': images,
            'label': labels}

    frame = pd.DataFrame(data, columns= ['image', 'label'])

    frame.to_csv(DATA_ANNOTATION)
    logger.info('Annotation created at {:s}'.format(DATA_ANNOTATION))

