import torch
import os
from os import sys, path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from .config import DATA_ANNOTATION

class RaspiDataset(Dataset):

    def __init__(self, annot, rootdir=os.getcwd(), transforms=None):

        self.data_root = rootdir
        self.transforms = transforms

        frame = pd.read_csv(annot)

        self.image_path_series = frame['image']
        self.labels = torch.tensor(frame['label'])

    def __getitem__(self, index):
        img_path = path.join(self.data_root, self.image_path_series[index])
        image = Image.open(img_path)
        return torch.tensor(image), self.labels

    def __len__(self):
        return len(self.labels)


def dataset():
    return RaspiDataset(DATA_ANNOTATION, rootdir=os.getcwd())


