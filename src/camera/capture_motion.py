from picamera import PiCamera
from gpiozero import MotionSensor
import os
from os import path
import sys
from threading import Thread, Lock

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(path.dirname(path.realpath(__file__)),'../..'))

from src.camera.capture import get_offset_index, get_path_format
from src.utils.logger import logger
from src.camera.cameraLED import CameraLED


def capture_motion(root_path, img_name_format, pir, camera, led, lock):
    i = get_offset_index(root_path, img_name_format, pir.pin)

    while True:
        try:
            pir.wait_for_motion()
            lock.acquire()
            led.on()
            full_img_path = path.join(root_path, img_name_format.format(pir.pin,i))
            camera.capture(full_img_path)
            os.system('play ../../ressources/klick.mp3')
            logger.info('picture \'{:s}\' taken.'.format(full_img_path))
            pir.wait_for_no_motion()
            i += 1
            led.off()
            lock.release()

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    pir0 = MotionSensor(17)
    pir1 = MotionSensor(27)

    camera = PiCamera(sensor_mode=2)

    led = CameraLED()
    led.off()

    img_name_format, root_path = get_path_format('unlabeled')
    img_name_format = 'pin{}_'+img_name_format

    lock = Lock()

    thread0 = Thread(target=capture_motion,
                     args=(root_path, img_name_format, pir0, camera, led, lock,))
    thread1 = Thread(target=capture_motion,
                     args=(root_path, img_name_format, pir1, camera, led, lock,))

    thread0.start()
    thread1.start()

    thread0.join()
    thread1.join()


