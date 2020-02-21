import cv2
import os
from src.camera.capture import get_offset_index, get_path_format
from src.utils.enums import Person
from src.utils.config import NUM_CLASSES


def label():
    person_list = [' {:s}: {:d} '.format(i.name, i.value) for i in Person]
    description = '|'.join(p for p in person_list) + '| Delete: d'

    window_name = 'LABEL ME!!!!1elf'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    img_name_format, root_path_src = get_path_format('unlabeled')

    for img_path in os.listdir(root_path_src):
        full_img_path = os.path.join(root_path_src, img_path)
        img = cv2.imread(full_img_path)

        imgcp = img.copy()
        imgcp = cv2.putText(imgcp, description, org=(0,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                            color=(0, 0, 255), thickness=2)

        cv2.imshow(window_name, imgcp)

        while True:
            key = cv2.waitKey(100)

            if key == 27:
                break

            # if class num gets over 10 have fun to find the right key from ascii table :)

            elif key in range(ord('0'), ord('0') + NUM_CLASSES):
                _, root_path_dst = get_path_format(Person(key - ord('0')).name.capitalize())
                i = get_offset_index(root_path_dst, img_name_format)
                new_img_path = os.path.join(root_path_dst, img_name_format.format(i))
                cv2.imwrite(new_img_path, img)
                os.remove(os.path.join(full_img_path))
                break

            elif key == ord('d'):
                os.remove(os.path.join(full_img_path))
                break

        if key == 27:
            break

    cv2.destroyAllWindows()
