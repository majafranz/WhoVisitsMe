from picamera import PiCamera
from gpiozero import MotionSensor
from os import path
import sys

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(path.dirname(path.realpath(__file__)),'../..'))

from src.camera.capture import get_offset_index, get_path_format
from src.utils.logger import logger

if __name__ == '__main__':
    pir = MotionSensor(4)
    camera = PiCamera(sensor_mode=2)

    img_name_format, root_path = get_path_format('unlabeled')
    i = get_offset_index(root_path, img_name_format)


    while True:
        try:
            pir.wait_for_motion()
            full_img_path = path.join(root_path, img_name_format.format(i))
            camera.capture(full_img_path)
            logger.info('picture \'{:s}\' taken.'.format(full_img_path))
            pir.wait_for_no_motion()
            i+=1

        except KeyboardInterrupt:
            break

