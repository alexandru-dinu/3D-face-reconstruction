import os

import numpy as np
import torch
from oct2py import octave

import data_loader
from utils import get_args, to_cuda
from visualize import visualize

# models here
from model_hourglass import StackedHourGlass


# ---


def get_coords(out, sz=0.25, thr=0.3):
    mat = np.sum(out >= thr, axis=0)

    pts = np.argwhere(mat >= thr)

    x, y = pts[:, 0], pts[:, 1]
    z = sz * mat[mat > 0].reshape(
        -1,
    )
    assert len(x) == len(y) == len(z)

    return x, y, z


def to_stl(file, x, y, z):
    # run octave
    octave.addpath(".")
    octave.stlwrite(f"../meshes/{file}.stl", x, y, z)


if __name__ == "__main__":
    # model output (voxel volume) -> obj -> meshlab visualization

    os.makedirs("../meshes", exist_ok=True)

    args = get_args()

    data = data_loader.FacesWith3DCoords(
        images_dir=args.images_dir,
        mats_dir=args.mats_dir,
        landmarks_dir=args.lands_dir,
        transform=args.transform,
    )

    model = StackedHourGlass(
        nChannels=224, nStack=2, nModules=2, numReductions=4, nOutputs=200
    )
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"[+] Loaded model {args.checkpoint}")

    i = np.random.randint(len(data))
    img, true_vol = data[i]
    print(f"[+] Loaded image {i}")

    img = to_cuda(img.unsqueeze(0), True)
    out = model(img)
    # assert len(out) == 1

    # D, H, W
    out = out[0].detach().squeeze().cpu().numpy()
    # print(out.shape)
    img = img.squeeze(0).cpu() + 128.0

    thr = 0.99
    x, y, z = get_coords(out, thr)
    to_stl("out", x, y, z)
    # --
    visualize(img, torch.from_numpy(out), thr=thr)
