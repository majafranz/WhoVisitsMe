from picamera import PiCamera
from gpiozero import MotionSensor
import os
# import torch
from os import path
import sys
from PIL import Image
import random
from threading import Thread, Lock
import tvm
import numpy as np

if __name__ == '__main__' and __package__ is None:
    sys.path.append(path.join(path.dirname(path.realpath(__file__)), '..'))

from src.core.model import model
from src.camera.capture import get_path_format, get_offset_index
from src.utils.logger import logger
from src.core.data import CustomTransforms
from src.utils.enums import Person
from src.optimize.load_model import load_module, input_name

def process_output(output):
    prediction = np.argmax(output, axis=1)[0]

    person = Person(prediction)
    
    if (person is not Person.NONE) and (person is not Person.OTHER):
        sound_root = os.path.join('..', 'ressources', person.name.capitalize())

        sound_sample = random.choice(os.listdir(sound_root))

        play_sound(os.path.join(sound_root, sound_sample))

    print('{:s}, du stinkst.'.format(person.name.capitalize()))


def play_sound(sound_path):
    os.system('play {:s}'.format(sound_path))


def detect_person(net, num_save, tmp_img_root, img_format, transforms, pir, camera, lock):
    i = get_offset_index(tmp_img_root, img_format, pir.pin)
    num_save += i
    while True:
        pir.wait_for_motion()
        lock.acquire()
        full_img_path = path.join(tmp_img_root, img_format.format(pir.pin, i))
        camera.capture(full_img_path)
        logger.info('picture \'{:s}\' taken.'.format(full_img_path))

        img = Image.open(full_img_path)
        img = transforms(img)
        img = img.unsqueeze(0)

        net.set_input(input_name, tvm.nd.array(img))  # net(img)
        net.run()
        output = net.get_output(0).asnumpy()
        print(output)        
        process_output(output)

        if i < num_save:
            i += 1
        else:
            i = 0

        pir.wait_for_no_motion()
        lock.release()


if __name__ == '__main__':
    pir0 = MotionSensor(17)
    pir1 = MotionSensor(27)

    camera = PiCamera(sensor_mode=2)

    tmp_img_root = path.join('..', 'TMP')
    os.makedirs(tmp_img_root, exist_ok=True)

    img_format, _ = get_path_format("")
    img_format = 'pin{}_' + img_format
    
    num_save = 5000
    lock = Lock()

    transforms = CustomTransforms(244, 1.3)
    # net, _, _, _ = model(device=torch.device('cpu'), load_path='prototype.pt')
    module = load_module()

    thread0 = Thread(target=detect_person,
                     args=(module, num_save, tmp_img_root, img_format, transforms, pir0, camera, lock,))
    thread1 = Thread(target=detect_person,
                     args=(module, num_save, tmp_img_root, img_format, transforms, pir1, camera, lock,))

    thread0.start()
    thread1.start()

    thread0.join()
    thread1.join()
