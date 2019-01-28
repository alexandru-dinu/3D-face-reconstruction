import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../face-alignment"))

from skimage import io
import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

img = io.imread(sys.argv[1])
preds = fa.get_landmarks(img)

for point in preds[0]:
    x, y = point
    cv2.circle(img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

plt.imshow(img)
plt.show()
