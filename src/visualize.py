import cv2
import numpy as np
import visvis as vv

import data_loader
from utils import get_args


def visualize(im, vol, sz=0.25, thr=0.99):
    im, vol = im.numpy(), vol.numpy()
    print("Image shape:", im.shape)
    print("Volume shape:", vol.shape)

    # overlap with 3d representation + BGR->RGB
    im = im.transpose(1, 2, 0)  # H,W,C
    im = im[:, :, ::-1]
    im = cv2.resize(im, (128, 128))

    t = vv.imshow(im)
    t.interpolate = True  # interpolate pixels

    # volshow will use volshow3 and rendering the isosurface if OpenGL version is >= 2.0
    # Otherwise, it will show slices with bars that you can move (much less useful).
    im = (im * 128 + 128).astype(np.uint8)
    # im = np.ones_like(im)

    volRGB = np.stack(
        (
            (vol >= thr) * im[:, :, 0],
            (vol >= thr) * im[:, :, 1],
            (vol >= thr) * im[:, :, 2],
        ),
        axis=3,
    )

    v = vv.volshow(volRGB, renderStyle="iso")
    v.transformations[1].sz = sz  # Z rescaling

    l0 = vv.gca()
    l0.light0.ambient = 0.9  # 0.2 is default for light 0
    l0.light0.diffuse = 1.0  # 1.0 is default

    a = vv.gca()
    a.axis.visible = 0
    a.camera.fov = 0  # orthographic
    vv.use().Run()


if __name__ == "__main__":
    args = get_args()

    trainset = data_loader.FacesWith3DCoords(
        images_dir=args.images_dir,
        mats_dir=args.mats_dir,
        landmarks_dir=args.lands_dir,
        transform=args.transform,
    )

    i = np.random.randint(len(trainset))
    img, volume, _ = trainset[i]
    visualize(img, volume)
