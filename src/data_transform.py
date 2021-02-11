from __future__ import division

import warnings

import cv2
import numpy as np

import data_loader
from utils import get_args

warnings.filterwarnings("ignore")


def rotate(img, alpha):
    """
    img.shape must be H,W,C
    alpha is in degrees
    """
    h, w = img.shape[:-1]
    M = cv2.getRotationMatrix2D((h / 2, w / 2), alpha, 1)
    return cv2.warpAffine(img, M, (h, w))


class Resize(object):
    def __call__(self, image, output_size):
        return cv2.resize(image, (output_size, output_size))


class Translation(object):
    def __call__(self, image, tx, ty):
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, image.shape[:-1])


class Rotation(object):
    def __call__(self, image, alpha):
        size, _, _ = image.shape
        return rotate(image, alpha)


class Flip(object):
    def __call__(self, image, mode):
        return cv2.flip(image, mode)


class Scale(object):
    def __call__(self, image, scale):
        size, _, _ = image.shape
        img = cv2.resize(image, None, fx=scale, fy=scale)
        new_size, _, _ = img.shape

        if scale > 1.0:
            offset = (new_size - size) // 2
            img = img[offset : offset + size, offset : offset + size, :]
            return img

        final_img = np.zeros(image.shape, dtype=np.uint8)
        offset = (size - new_size) // 2

        final_img[offset : offset + new_size, offset : offset + new_size, :] = img
        return final_img
