import os

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

import data_transform
from utils import get_args, gaussian_distribution


class FacesWith3DCoords(Dataset):
    def __init__(
        self,
        images_dir: str,
        mats_dir: str,
        landmarks_dir: str = None,
        transform: bool = False,
    ):
        self.images, self.volumes, self.landmarks = [], [], []
        self.transform = transform

        if transform:
            self.tf_flip = data_transform.Flip()
            self.tf_rotate = data_transform.Rotation()
            self.tf_translate = data_transform.Translation()
            self.tf_scale = data_transform.Scale()

        for i in os.listdir(images_dir):
            name = i.split(".")[0]

            self.images += [os.path.join(images_dir, name + ".jpg")]
            self.volumes += [os.path.join(mats_dir, name + ".mat")]

            if landmarks_dir:
                self.landmarks += [os.path.join(landmarks_dir, name + ".mat")]

        assert len(self.images) == len(self.volumes)

    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        # img is H,W,C
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        size, _, _ = img.shape

        # construct gaussians for each landmark
        lands = np.zeros((68, size, size))
        if self.landmarks:
            x_lands, y_lands = scipy.io.loadmat(self.landmarks[index])["pt2d"].astype(
                np.int32
            )
            for i in range(len(x_lands)):
                lands[i] = gaussian_distribution(x_lands[i], y_lands[i], size)
                lands[i] = data_transform.rotate(np.expand_dims(lands[i], axis=2), 90)
            lands = np.transpose(lands, axes=(1, 2, 0))

        # load 3D coordinates
        x, y, z = scipy.io.loadmat(self.volumes[index])["Fitted_Face"].astype(np.int32)
        z = z - z.min()

        depth_map = np.zeros((size, size), dtype=np.float)
        for i in range(len(x)):
            depth_map[x[i], y[i]] = max(z[i], depth_map[x[i], y[i]])

        depth_map = cv2.morphologyEx(
            depth_map,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        depth_map = cv2.GaussianBlur(depth_map, ksize=(5, 5), sigmaX=3, sigmaY=3)
        depth_map = data_transform.rotate(np.expand_dims(depth_map, axis=2), 90)

        # binary volume
        volume = np.zeros((size, size, 200), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                volume[i, j, : int(depth_map[i, j])] = 1

        # data augmentation (prob 0.2)
        if self.transform and np.random.rand() < 0.2:
            img, volume = self.tf_flip(img, 1), self.tf_flip(volume, 1)
            if self.landmarks:
                lands = self.tf_flip(lands, 1)

            alpha = np.random.randint(-45, 45)
            tx, ty = np.random.randint(-15, 15), np.random.randint(-15, 15)
            factor = 0.85 + np.random.rand() * (1.15 - 0.85)

            img, volume = self.tf_rotate(img, alpha), self.tf_rotate(volume, alpha)
            img, volume = self.tf_translate(img, tx, ty), self.tf_translate(
                volume, tx, ty
            )
            img, volume = self.tf_scale(img, factor), self.tf_scale(volume, factor)

            if self.landmarks:
                lands = self.tf_rotate(lands, alpha)
                lands = self.tf_translate(lands, tx, ty)
                lands = self.tf_scale(lands, factor)

        R = data_transform.Resize()
        img, volume, lands = R(img, 128), R(volume, 128), R(lands, 128)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        volume = torch.from_numpy(volume.transpose(2, 0, 1))
        lands = torch.from_numpy(lands.transpose(2, 0, 1))

        img = (img - 128) / 128.0

        # C, H, W
        return img, volume, lands

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    args = get_args()

    data = FacesWith3DCoords(
        images_dir=args.images_dir,
        mats_dir=args.mats_dir,
        landmarks_dir=args.lands_dir,
        transform=args.transform,
    )

    from visualize import visualize

    for idx in range(20):
        i, m, l = data[np.random.randint(len(data))]

        visualize(i, m, sz=0.25, thr=0.99)
