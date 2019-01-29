import numpy as np
import torch

import dataloaders
from utils import get_args, to_cuda
from visualize import visualize
from model_vgg import VGGModel

if __name__ == '__main__':
    args = get_args()
    args.images_dir = "../300W-3D-all/images"
    args.mats_dir = "../300W-3D-all/3d-scans"
    args.lands_dir = "../300W-3D-all/landmarks"
    args.checkpoint = "/home/robert/PycharmProjects/3D-face-reconstruction/checkpoints/simple_model_60"

    data = dataloaders.FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, lands_dir=args.lands_dir, transform=args.transform
    )

    model = VGGModel(out_size=50 * 50)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"Loaded model {args.checkpoint}")

    i = np.random.randint(len(data))
    img, _ = data[i]

    img = to_cuda(img.unsqueeze(0), True)
    # out = F.sigmoid(model(img))
    out = model(img).detach().squeeze(0).cpu().numpy().reshape((50, 50))
    print(out.shape, type(out), out.max())

    mat = np.zeros((50, 50, 200), dtype=np.uint8)
    for i in range(50):
        for j in range(50):
            mat[i, j, :int(out[i, j])] = 1

    img = img.squeeze(0).cpu()
    # out = out.detach().squeeze(0).cpu()
    import datatransform
    import cv2

    mat = datatransform.Resize()(mat, 200)
    mat = cv2.GaussianBlur(mat, (5, 5), sigmaX=7, sigmaY=7)

    # visualize(img, out)
    visualize(img[:3, :, :], torch.from_numpy(mat.transpose((2, 0, 1))))
