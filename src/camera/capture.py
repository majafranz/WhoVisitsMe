from picamera import PiCamera
from time import sleep
import os
from os import sys, path
import shutil

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(os.path.dirname(os.path.realpath(__file__)),'../..'))

from src.utils.config import DATA_ROOT, PERSON, IMG_NUM, TIME_DISTANCE, RESOLUTION, TIME_OFFSET, RESET

def timer():
    for i in range(TIME_OFFSET):
        print(TIME_OFFSET-i)
        sleep(1)

# create destination folders and calculate image offset index
def get_offset_index(root_path, img_name_format):
    offset = 0
    if path.exists(root_path):
        if RESET:
            shutil.rmtree(root_path)
            os.makedirs(root_path, exist_ok=True)

        else:
            while os.path.isfile(path.join(root_path, img_name_format.format(offset))):
                offset += 1
    else:
        os.makedirs(root_path, exist_ok=True)

    return offset


if __name__ == '__main__':
    camera = PiCamera(sensor_mode=2)

    if RESOLUTION is not None:
        camera.resolution = RESOLUTION

    dirname = str(PERSON).split('.')[1].capitalize()
    root_path = path.join(DATA_ROOT, dirname)
    img_name_format = '{:06d}.jpg'

    offset = get_offset_index(root_path, img_name_format)
    camera.start_preview()
    timer()

    for i in range(offset, IMG_NUM+offset):
        full_img_path = path.join(root_path, img_name_format.format(i))
        sleep(TIME_DISTANCE)
        camera.capture(full_img_path)

    camera.stop_preview()

