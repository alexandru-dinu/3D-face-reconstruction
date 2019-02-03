import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../face-alignment"))

import face_alignment

net_face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def get_lands(img):
    _img = img.copy()

    preds = net_face_align.get_landmarks(_img)

    for point in preds[0]:
        x, y = point
        cv2.circle(_img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    return _img


if __name__ == '__main__':
    lands = get_lands(cv2.imread(sys.argv[1]))

    plt.imshow(lands)
    plt.show()
