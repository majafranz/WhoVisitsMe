from src.utils.enums import Person
from os import path

##### - OVERALL CONFIG - #####
REL_PATH = path.join('..')  # easy to change relative paths for root folder

##### - CAPTURE CONFIG - #####
PERSON = Person.MAJA
IMG_NUM = 100
TIME_DISTANCE = 1  # time between two captures in s
TIME_OFFSET = 10  # seconds before capturing starts
RESOLUTION = None
RESET = False

##### - DATA CONFIG - #####
DATA_ROOT = path.join(REL_PATH, 'data')  # done with join due to windows using \ and linux /
DATA_ANNOTATION = path.join(DATA_ROOT, "annotation.csv")

IMAGE_SIZE = 244  # size of input image
IMAGE_SCALE = 1.3  # how much of the image can be cut off

BATCH_SIZE = 12
NUM_WORKERS = 0
SHUFFLE = True
TRAIN_SPLIT = 0.8
RANDOM_SEED = False  # If true it uses different training test split for each training. I'm not sure if it is good tbh

##### - TRAINING CONFIG - #####
NUM_CLASSES = 5
NUM_EPOCHS = 10
PLOT = True
LR = 0.001
SAVE_PATH = path.join(REL_PATH, 'models')
LOAD_MODEL_PATH = 'test.pt'
SPEC_SAVE_NAME = None

##### - ADDITIONAL CONFIG - #####
LOG_ROOT = path.join(REL_PATH, 'log')
