from picamera import PiCamera
from gpiozero import MotionSensor
import os
import torch
from os import path
import sys
from PIL import Image

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(path.dirname(path.realpath(__file__)), '..'))

from src.core.model import model
from src.camera.capture import get_path_format
from src.utils.logger import logger
from src.core.data import CustomTransforms
from src.utils.enums import Person


def process_output(output):
    _, pred = torch.max(output, 1)
    prediction = pred.item()

    person = Person(prediction).name.capitalize()

    print('{:s}, du stinkst.')


if __name__ == '__main__':
    net = model('prototype.pt')
    transforms = CustomTransforms
    pir = MotionSensor(4)
    camera = PiCamera(sensor_mode=2)
    tmp_img_root = path.join('..', 'TMP')
    img_format, _ = get_path_format("")

    os.makedirs(tmp_img_root)
    num_save = 10
    i = 0
    while True:
        try:
            pir.wait_for_motion()
            full_img_path = path.join(tmp_img_root, img_format.format(i))
            camera.capture(full_img_path)
            logger.info('picture \'{:s}\' taken.'.format(full_img_path))

            img = Image.open(full_img_path)
            img = transforms(img)
            img = img.unsqueeze(0)

            output = net(img)
            process_output(output)

            pir.wait_for_no_motion()

            if i < num_save:
                i += 1
            else:
                i = 0

        except KeyboardInterrupt:
            break

