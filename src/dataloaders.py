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


    def __init__(self, images_dir: str, mats_dir: str, transform: bool = False):
        self.images, self.mats, self.lands = [], [], []
        self.transform = transform

        for i in os.listdir(images_dir)[:10]:
            if i.endswith(".jpg"):
                self.images.append(os.path.join(images_dir, i))
                self.mats.append(os.path.join(mats_dir, i.split(".")[0] + ".mat"))
                self.lands.append(os.path.join(mats_dir, i.split(".")[0] + "_landmark.mat"))

        assert len(self.images) == len(self.mats)


    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        # img is H,W,C
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        size, _, _ = img.shape

        lands = []
        x_lands, y_lands = scipy.io.loadmat(self.lands[index])['pt2d'].astype(np.int32)
        for i in tqdm(range(len(x_lands))):
            lands.append(gaussian_distribution(x_lands[i], y_lands[i]))
        lands = np.array(lands)

        x, y, z = scipy.io.loadmat(self.mats[index])['Fitted_Face'].astype(np.int32)
        z = z - z.min()

        gray = np.zeros((size, size), dtype=np.float)
        for i in range(len(x)):
            gray[x[i], y[i]] = z[i]
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=3, sigmaY=3)
        gray = datatransform.rotate(np.expand_dims(gray, axis=2), alpha=90)

        mat = np.zeros((size, size, 200), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                mat[i, j, :int(gray[i, j])] = 1

        if np.random.rand() < 0.2:
            flip = datatransform.Flip()
            # for visualization axis are flipped
            img, mat = flip(img, 1), flip(mat, 1)

        if np.random.rand() < 0.2 and self.transform:
            alpha = np.random.randint(-45, 45)
            tx, ty = np.random.randint(-15, 15), np.random.randint(-15, 15)
            factor = 0.85 + np.random.rand() * (1.15 - 0.85)

            rot = datatransform.Rotation()
            trans = datatransform.Translation()
            scale = datatransform.Scale()

            img, mat = rot(img, alpha), rot(mat, alpha)
            # for visualization axis are flipped
            img, mat = trans(img, tx, ty), trans(mat, tx, ty)
            img, mat = scale(img, factor), scale(mat, factor)

        # resize image to 200 x 200 and mat to 192x192
        R = datatransform.Resize()
        img, mat = R(img, 200), R(mat, 184)

        # C, H, W
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(mat.transpose(2, 0, 1)), torch.from_numpy(lands)


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    args = get_args()
    args.images_dir = "/home/nemodrive2/dan_m/3D-face-reconstruction/src/300W-3D/AFW"
    args.mats_dir = "/home/nemodrive2/dan_m/3D-face-reconstruction/src/300W-3D/AFW"
    args.lands_dir = "/home/nemodrive2/dan_m/3D-face-reconstruction/src/300W-3D/AFW"

    d = FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    )

    i, m, lands = d[np.random.randint(len(d))]
    # print(m)

    cv2.imshow("Image", i.numpy().transpose(1, 2, 0))
    cv2.waitKey(0)
