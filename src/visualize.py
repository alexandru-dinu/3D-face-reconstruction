import cv2
import numpy as np
import visvis as vv

import dataloaders
from utils import get_args


def visualize(im, vol):
    # convert im and vol to numpy
    im, vol = im.numpy(), vol.numpy()
    print("Image shape:", im.shape)
    print("Volume shape:", vol.shape)

    # overlap with 3d representation + BGR->RGB
    im = im.transpose(1, 2, 0)  # H,W,C
    im = im[:, :, ::-1]

    # resize image to 192 x 192
    im = cv2.resize(im, (192, 192))

    t = vv.imshow(im)
    t.interpolate = True  # interpolate pixels

    # volshow will use volshow3 and rendering the isosurface if OpenGL version is >= 2.0
    # Otherwise, it will show slices with bars that you can move (much less useful).
    im = np.ones_like(im)
    volRGB = np.stack(((vol >= 0.5) * im[:, :, 0],
                       (vol >= 0.5) * im[:, :, 1],
                       (vol >= 0.5) * im[:, :, 2]), axis=3)

    v = vv.volshow(volRGB, renderStyle='iso')
    v.transformations[1].sz = 0.5  # Z was twice as deep during training

    l0 = vv.gca()
    l0.light0.ambient = 0.9  # 0.2 is default for light 0
    l0.light0.diffuse = 1.0  # 1.0 is default

    a = vv.gca()
    a.camera.fov = 0  # orthographic
    vv.use().Run()


if __name__ == "__main__":
    args = get_args()
    args.images_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"
    args.mats_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"

    trainset = dataloaders.FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    )

    i = np.random.randint(len(trainset))
    im, vol = trainset[i]
    visualize(im, vol)
