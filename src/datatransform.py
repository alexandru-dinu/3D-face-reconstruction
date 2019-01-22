from __future__ import division
import sys
import os
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import dataloaders

import warnings
warnings.filterwarnings("ignore")


class Resize(object):
    def __call__(self, image, output_size):
        return cv2.resize(image, (output_size, output_size))


class Translation(object):
    def __call__(self, image, tx, ty):
        # get translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, image.shape[:-1])


class Rotation(object):
    def __call__(self, image, angle):
        # get rotation matrix
        size, _, _ = image.shape
        alpha = np.random.randint(-45, 45)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), alpha, 1)
        return cv2.warpAffine(image, M, image.shape[:-1])


class FlipHorizontal(object):
    def __call__(self, image):
        return cv2.flip(image, 1)


class Scale(object):
    def __call__(self, image, scale):
        size, _, _ = image.shape
        img = cv2.resize(image, None, fx=scale, fy=scale)
        new_size, _, _ = img.shape

        if scale > 1.0:
            offset = (new_size - size) // 2
            img = img[offset:offset + size, offset:offset + size, :]
            return img

        final_img = np.zeros(image.shape, dtype=np.uint8)
        offset = (size - new_size) // 2

        final_img[offset:offset + new_size, offset:offset + new_size, :] = img
        return final_img



if __name__ == "__main__":
    d = dataloaders.FacesWith3DCoords(images_dir=sys.argv[1], mats_dir=sys.argv[2])
    img2D, img3D = d[np.random.randint(len(d))]

    # Transf = Translation()
    # Transf = Resize(192)
    Transf = FlipHorizontal()
    # Transf = Scale()
    timg2D = Transf(img2D.numpy().transpose(1, 2, 0))

    print(timg2D.shape)

    cv2.imshow("Image", timg2D)
    cv2.waitKey(0)

    cv2.imshow("Orig image", img2D.numpy().transpose(1, 2, 0))
    cv2.waitKey(0)
