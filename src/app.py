from picamera import PiCamera
from gpiozero import MotionSensor
import os
import torch
from os import path
import sys
from PIL import Image
import random
from threading import Thread, Lock

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
    person = Person(prediction)
    
    sound_root = os.path.join('..','ressources', person.name.capitalize())
    
    sound_sample = random.choice(os.listdir(sound_root))

    play_sound(os.path.join(sound_root, sound_sample))

    print('{:s}, du stinkst.'.format(person.name.capitalize()))


def play_sound(sound_path):
    os.system('play {:s}'.format(sound_path))

def detect_person(net, pir, camera, lock):
    transforms = CustomTransforms(244, 1.3)

    tmp_img_root = path.join('..', 'TMP')
    img_format, _ = get_path_format("")
    
    img_format = 'pin{}_' + img_format

    os.makedirs(tmp_img_root, exist_ok=True)

    num_save = 1000
    i = 0

    while True:
        pir.wait_for_motion()
        lock.acquire()
        full_img_path = path.join(tmp_img_root, img_format.format(pir.pin,i))
        camera.capture(full_img_path)
        logger.info('picture \'{:s}\' taken.'.format(full_img_path))

        img = Image.open(full_img_path)
        img = transforms(img)
        img = img.unsqueeze(0)

        output = net(img)
        print(output)
        process_output(output)

        pir.wait_for_no_motion()
        lock.release()

        if i < num_save:
            i += 1
        else:
            i = 0


if __name__ == '__main__':
    pir0 = MotionSensor(17)
    pir1 = MotionSensor(27)
    camera = PiCamera(sensor_mode=2)
    lock = Lock()
    net, _, _, _ = model(device=torch.device('cpu'), load_path='prototype.pt')
    thread0 = Thread(target=detect_person, args=(net, pir0, camera,lock,))
    thread1 = Thread(target=detect_person, args=(net, pir1, camera,lock,))
    thread0.start()
    thread1.start()
    thread0.join()
    thread1.join()
