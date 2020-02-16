from src.utils.enums import Person
from os import path
from torchvision import transforms as T

##### - CAPTURE CONFIG - #####
PERSON = Person.MAJA
IMG_NUM = 100
TIME_DISTANCE = 1  # time between two captures in s
TIME_OFFSET = 10  # seconds before capturing starts
RESOLUTION = None
RESET = True

##### - DATA CONFIG - #####
DATA_ROOT = path.join('..', '..', 'data')  # done with join due to windows using \ and linux /
DATA_ANNOTATION = path.join(DATA_ROOT, "annotation.csv")

IMAGE_SIZE = 244  # size of input image
IMAGE_SCALE = 1.3  # how much of the image can be cut off

BATCH_SIZE = 4
NUM_WORKERS = 5
SHUFFLE = True
TRAIN_SPLIT = 0.8

##### - TRAINING CONFIG - #####
NUM_CLASSES = 2

##### - ADDITIONAL CONFIG - #####
LOG_ROOT = path.join('..', '..', 'log')
