import cv2
import os
from src.camera.capture import get_offset_index, get_path_format
from src.utils.enums import Person

if __name__=='__main__':
    cv2.namedWindow('LABEL ME!!!!1elf')

    img_name_format, root_path_src = get_path_format('unlabeled')

    for img_path in os.listdir(root_path_src):
        full_img_path = os.path.join(root_path_src, img_path)
        img = cv2.imread(full_img_path)
        cv2.imshow('LABEL ME!!!!1elf', img)

        while True:
            key = cv2.waitKey(100)
            print(key)

            if key == 27:
                break

            elif key in range(ord('0'), ord('5')):
                _, root_path_dst = get_path_format(Person(key-ord('0')).name.capitalize())
                i = get_offset_index(root_path_dst, img_name_format)
                new_img_path = os.path.join(root_path_dst, img_name_format.format(i))
                cv2.imwrite(new_img_path, img)
                os.remove(os.path.join(full_img_path))
                break

        if key == 27:
            break

    cv2.destroyAllWindows()
