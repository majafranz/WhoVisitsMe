from picamera import PiCamera
from time import sleep
import os
from os import sys, path

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(os.path.dirname(os.path.realpath(__file__)),'../..'))

from src.utils.config import DATA_ROOT, PERSON, IMG_NUM, TIME_DISTANCE, RESOLUTION

if __name__ == '__main__':
    camera = PiCamera(sensor_mode=2)

    if RESOLUTION is not None:
         camera.resolution = RESOLUTION

    foldername = str(PERSON).split('.')[1].capitalize()
    img_name_format = '{:06d}'
    root_path = path.join(DATA_ROOT, foldername)

    # check if folders in path exist and if not create them
    os.makedirs(root_path, exist_ok=True)

    img_name_format = '{:06d}.jpg'

    camera.start_preview()
    for i in range(IMG_NUM):
        full_img_path = path.join(root_path, img_name_format.format(i))
        sleep(TIME_DISTANCE)
        camera.capture(full_img_path)

    camera.stop_preview()
