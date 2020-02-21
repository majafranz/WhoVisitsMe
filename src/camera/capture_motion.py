from picamera import PiCamera
from gpiozero import MotionSensor
from os import path

from src.camera.capture import get_offset_index
from src.utils.config import DATA_ROOT

if __name__ == '__main__':
    pir = MotionSensor(4)
    camera = PiCamera(sensor_mode=2)

    img_name_format = '{:06d}.jpg'
    dirname = 'unlabeled'
    root_path = path.join(DATA_ROOT, dirname)
    i = get_offset_index(root_path, img_name_format)


    while True:
        try:
            pir.wait_for_motion()
            full_img_path = path.join(root_path, img_name_format.format(i))
            camera.capture(full_img_path)
            pir.wait_for_no_motion()

        except KeyboardInterrupt:
            break

