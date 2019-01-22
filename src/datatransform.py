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
    """Resize the image in a sample to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        # image2D, image3D, landmarks = sample['image2D'], sample['image3D'], sample['landmarks']
        image2D, image3D = sample
        size, _, depth = image3D.shape

        # resize images
        img2D = cv2.resize(image2D, (self.output_size, self.output_size))
        img3D = cv2.resize(image3D, (self.output_size - 8, self.output_size - 8))

        # scale z component
        #img3D[-1] *= self.output_size / size

        # scale landmarks
        # landmarks = landmarks * [self.output_size / size, self.output_size / size]
        # return {'image2D': img2D, 'image3D': img3D, 'landmarks': landmarks}
        return img2D, img3D


class Translation(object):
    def __call__(self, sample):
        # image2D, image3D, landmarks = sample['image2D'], sample['image3D'], sample['landmarks']
        image2D, image3D = sample
        size, _, depth = image3D.shape

        # get translation matrix
        tx = np.random.randint(-15, 15)
        ty = np.random.randint(-15, 15)
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        img2D = cv2.warpAffine(image2D, M, image2D.shape[:-1])
        img3D = cv2.warpAffine(image3D, M, image3D.shape[:-1])

        # scale landmarks
        # landmarks = landmarks + [tx, ty]
        # return {'image2D': img2D, 'image3D': img3D, 'landmarks': landmarks}
        return img2D, img3D


class Rotation(object):
    def __call__(self, sample):
        # image2D, image3D, landmarks = sample['image2D'], sample['image3D'], sample['landmarks']
        image2D, image3D = sample
        size, _, depth = image3D.shape

        # get rotation matrix
        alpha = np.random.randint(-45, 45)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), alpha, 1)

        img2D = cv2.warpAffine(image2D, M, image2D.shape[:-1])
        img3D = cv2.warpAffine(image3D, M, image3D.shape[:-1])

        # scale landmarks
        # landmarks = M.dot(landmarks)
        # return {'image2D': img2D, 'image3D': img3D, 'landmarks': landmarks}
        return img2D, img3D


class FlipHorizontal(object):
    def __call__(self, sample):
        image2D, image3D = sample
        size, _, depth = image3D.shape

        img2D = cv2.flip(image2D, 1)
        img3D = cv2.flip(image3D, 1)

        return img2D, img3D


class Scale(object):
    def __call__(self, sample):
        image2D, image3D = sample
        size, _, depth = image3D.shape

        scale = 0.85 + np.random.rand(1) * (1.15 - 0.85)
        img2D = cv2.resize(image2D, None, fx=scale, fy=scale)
        img3D = cv2.resize(image3D, None, fx=scale, fy=scale)
        new_size, _, _ = img3D.shape

        if scale > 1.0:
            offset = (new_size - size) // 2
            img2D = img2D[offset:offset + size, offset:offset + size, :]
            img3D = img3D[offset:offset + size, offset:offset + size, :]
            return img2D, img3D

        final_img2D = np.zeros(image2D.shape, dtype=np.uint8)
        final_img3D = np.zeros(image3D.shape, dtype=np.uint8)
        offset = (size - new_size) // 2

        final_img2D[offset:offset + new_size, offset:offset + new_size, :] = img2D
        final_img3D[offset:offset + new_size, offset:offset + new_size, :] = img3D
        return final_img2D, final_img3D


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image2D, image3D, landmarks = sample['image2D'], sample['image3D'], sample['landmarks']
        image2D, image3D = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image2D = image2D.transpose((2, 0, 1)).astype(np.float32)
        image2D /= 255.0

        # return {'image2D': torch.from_numpy(image2D),
        #        'image3D': torch.from_numpy(image3D),
        #        'landmarks': torch.from_numpy(landmarks)}
        return torch.from_numpy(image2D), torch.from_numpy(image3D)


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    d = dataloader.FacesWith3DCoords(images_dir=sys.argv[1], mats_dir=sys.argv[2])
    img2D, img3D = d[np.random.randint(len(d))]


    # Transf = Translation()
    # Transf = Resize(192)
    # Transf = FlipHorizontal()
    Transf = Scale()
    timg2D, timg3D = Transf((img2D, img3D))

    print(timg2D.shape)

    cv2.imshow("Image", timg2D)
    cv2.waitKey(0)

    cv2.imshow("Orig image", img2D)
    cv2.waitKey(0)
