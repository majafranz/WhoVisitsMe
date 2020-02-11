from picamera import PiCamera
from time import sleep
import os
from ..core.config import DATA_ROOT, PERSON, IMG_NUM, TIME_DISTANCE

if __name__ == '__main__':
    camera = PiCamera()

    foldername = str(PERSON).split('.')[1].capitalize()
    img_name_format = '{:06d}'
    root_path = os.path.join(DATA_ROOT, foldername)

    # check if folders in path exist and if not create them
    os.makedirs(path, exist_ok=True)

    img_name_format = '{:06d}'

    camera.start_preview()
    for i in range(IMG_NUM):
        full_img_path = os.path.join(root_path, img_name_format.format(i))
        sleep(TIME_DISTANCE)
        camera.capture(full_img_path)

    camera.stop_preview()
