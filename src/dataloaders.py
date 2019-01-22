import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2
from torch.utils.data import Dataset
import datatransform
import torch


class FacesWith3DCoords(Dataset):
    """
    Images are in 300W-3D/<name>/*.jpg
    Coords are in 300W-3D-Face/<name>/*.mat
    """

    def __init__(self, images_dir: str, mats_dir: str, transform):
        self.images, self.mats = [], []
        self.transform = transform

        for i in os.listdir(images_dir):
            if i.endswith(".jpg"):
                self.images.append(os.path.join(images_dir, i))
                self.mats.append(os.path.join(mats_dir, i.split(".")[0] + ".mat"))

        assert len(self.images) == len(self.mats)

    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        size, _, _ = img.shape

        x, y, z = scipy.io.loadmat(self.mats[index])['Fitted_Face']
        z = z - z.min()

        x = x.astype(np.int32)
        y = y.astype(np.int32)
        z = z.astype(np.int32)

        mat = np.zeros((size, size, 200), dtype=np.uint8)
        for i in range(len(x)):
            mat[x[i], y[i], :z[i]] = 1

        # resize to 200 x 200
        R = datatransform.Resize(200)
        img, mat = R((img, mat))
        img = img.astype(np.float32)

        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(mat.transpose(2, 0, 1))

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import sys

    d = FacesWith3DCoords(images_dir=sys.argv[1], mats_dir=sys.argv[2])

    i, m = d[np.random.randint(len(d))]
    print(m)

    cv2.imshow("Image", i)
    cv2.waitKey(0)
