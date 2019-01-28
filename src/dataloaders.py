import os

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

import datatransform
from utils import get_args


class FacesWith3DCoords(Dataset):
    """
    Images are in 300W-3D/<name>/*.jpg
    Coords are in 300W-3D-Face/<name>/*.mat
    """

    def __init__(self, images_dir: str, mats_dir: str, transform: bool = False):
        self.images, self.mats = [], []
        self.transform = transform

        for i in os.listdir(images_dir):
            if i.endswith(".jpg"):
                self.images.append(os.path.join(images_dir, i))
                self.mats.append(os.path.join(mats_dir, i.split(".")[0] + ".mat"))

        assert len(self.images) == len(self.mats)

    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        # img is H,W,C
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        size, _, _ = img.shape

        x, y, z = scipy.io.loadmat(self.mats[index])['Fitted_Face'].astype(np.int32)
        z = z - z.min()

        gray = np.zeros((size, size), dtype=np.float)
        for i in range(len(x)):
            gray[x[i], y[i]] = z[i]
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)


        mat = np.zeros((size, size, 200), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                mat[i, j, :int(gray[i, j])] = 1

        if np.random.rand() < 0.2:
            flip = datatransform.Flip()
            # for visualization axis are flipped
            img, mat = flip(img, 1), flip(mat, 0)

        if np.random.rand() < 0.2 and self.transform:
            alpha = np.random.randint(-45, 45)
            tx, ty = np.random.randint(-15, 15), np.random.randint(-15, 15)
            factor = 0.85 + np.random.rand() * (1.15 - 0.85)

            rot = datatransform.Rotation()
            trans = datatransform.Translation()
            scale = datatransform.Scale()

            img, mat = rot(img, alpha), rot(mat, alpha)
            # for visualization axis are flipped
            img, mat = trans(img, tx, ty), trans(mat, -ty, tx)
            img, mat = scale(img, factor), scale(mat, factor)

        # resize image to 200 x 200 and mat to 192x192
        R = datatransform.Resize()
        img, mat = R(img, 200), R(mat, 192)

        # C, H, W
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(mat.transpose(2, 0, 1))

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    args = get_args()
    args.images_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"
    args.mats_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"

    d = FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    )

    i, m = d[np.random.randint(len(d))]
    # print(m)

    cv2.imshow("Image", i.numpy().transpose(1, 2, 0))
    cv2.waitKey(0)
