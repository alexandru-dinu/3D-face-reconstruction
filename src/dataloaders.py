import os

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

import datatransform
from utils import get_args, gaussian_distribution
from tqdm import tqdm


class FacesWith3DCoords(Dataset):
    """
    Images are in 300W-3D/<name>/*.jpg
    Coords are in 300W-3D-Face/<name>/*.mat
    """


    def __init__(self, images_dir: str, mats_dir: str, lands_dir: str, transform: bool = False):
        self.transform = transform
        self.images, self.mats, self.lands = [], [], []

        for img in sorted(os.listdir(images_dir))[:100]:
            name = img.split(".")[0]

            self.images += [os.path.join(images_dir, name + ".jpg")]
            self.mats += [os.path.join(mats_dir, name + ".mat")]
            self.lands += [os.path.join(lands_dir, name + ".mat")]

        assert len(self.images) == len(self.mats) == len(self.lands)


    def __getitem__(self, index):
        assert 0 <= index < len(self.images)
        # img is H,W,C
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        size, _, _ = img.shape

        # read landmarks
        lands = np.zeros((68, size, size))
        x_lands, y_lands = scipy.io.loadmat(self.lands[index])['pt2d'].astype(np.int32)
        for i in range(len(x_lands)):
            lands[i] = gaussian_distribution(x_lands[i], y_lands[i], size)
        lands = np.transpose(lands, (1, 2, 0))
        lands = datatransform.rotate(lands, -90)

        # read 3D points
        x, y, z = scipy.io.loadmat(self.mats[index])['Fitted_Face'].astype(np.int32)
        z = z - z.min()

        gray = np.zeros((size, size), dtype=np.float)
        for i in range(len(x)):
            gray[x[i], y[i]] = z[i]
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=3, sigmaY=3)
        gray = datatransform.rotate(np.expand_dims(gray, axis=2), alpha=90)

        mat = np.zeros((size, size, 200), dtype=np.uint8)
        # for i in range(size):
        #    for j in range(size):
        #        mat[i, j, :int(gray[i, j])] = 1

        if np.random.rand() < -1.2 and self.transform:
            flip = datatransform.Flip()
            # for visualization axis are flipped
            # img, mat = flip(img, 1), flip(mat, 1)
            img, gray, lands = flip(img, 1), flip(np.expand_dims(gray, axis=2), 1), flip(lands, 1)

        if np.random.rand() < -1.2 and self.transform:
            alpha = np.random.randint(-45, 45)
            tx, ty = np.random.randint(-15, 15), np.random.randint(-15, 15)
            factor = 0.85 + np.random.rand() * (1.15 - 0.85)

            rot = datatransform.Rotation()
            trans = datatransform.Translation()
            scale = datatransform.Scale()

            # img, mat = rot(img, alpha), rot(mat, alpha)
            # img, mat = trans(img, tx, ty), trans(mat, tx, ty)
            # img, mat = scale(img, factor), scale(mat, factor)
            img, gray, lands = rot(img, alpha), rot(np.expand_dims(gray, axis=2), alpha), rot(lands, alpha)
            img, gray, lands = trans(img, tx, ty), trans(np.expand_dims(gray, axis=2), tx, ty), trans(lands, tx, ty)
            img, gray, lands = scale(img, factor), scale(np.expand_dims(gray, axis=2), factor), scale(lands, factor)

        # resize image to 200 x 200 and mat to 192x192
        R = datatransform.Resize()

        # img, mat = R(img, 200), R(mat, 184)
        gray = np.expand_dims(gray, axis=2)
        # print(img.shape, gray.shape, lands.shape)
        img, gray, lands = R(img, 224), R(gray, 50), R(lands, 200)
        gray = np.expand_dims(gray, axis=2)
        # gray = np.zeros_like(gray)

        # C, H, W
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(gray.transpose(2, 0, 1))  # , torch.from_numpy(lands.transpose(2, 0, 1))
        # img = np.concatenate([img, lands], axis=2)
        # return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(gray.transpose(2, 0, 1))


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    args = get_args()
    args.images_dir = "/home/robert/PycharmProjects/3D-face-reconstruction/300W-3D-all/images"
    args.mats_dir = "/home/robert/PycharmProjects/3D-face-reconstruction/300W-3D-all/3d-scans"
    args.lands_dir = "/home/robert/PycharmProjects/3D-face-reconstruction/300W-3D-all/landmarks"

    d = FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, lands_dir=args.lands_dir, transform=args.transform
    )

    i, m, l = d[0]  # d[np.random.randint(len(d))]
    # i, m, lands = d[np.random.randint(len(d))]
    # print(m)
    import matplotlib.pyplot as plt

    # plt.imshow(l.numpy().sum(axis=0))
    # plt.show()

    l3 = np.zeros((3, 450, 450))
    l3 += l.numpy().sum(axis=0)
    l3 = np.transpose(l3, (1, 2, 0))

    plt.imshow(l3 * 10 + i.numpy().transpose(1, 2, 0) / 255.0)
    plt.show()
