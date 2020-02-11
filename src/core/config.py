from os import sys, path
if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from ..utils.enums import Person

##### - CAPTURE CONFIG - #####
PERSON = Person.MAJA
IMG_NUM = 100
TIME_DISTANCE = 1 # time between two captures in s

##### - DATA CONFIG - #####
DATA_ROOT = "../../data"
DATA_ANNOTATION = ""

##### - TRAINING CONFIG - #####
NUM_CLASSES = 2
