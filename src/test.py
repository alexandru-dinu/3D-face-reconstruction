import numpy as np
import torch
import torch.nn.functional as F

import dataloaders
from u_net import UNet, to_cuda
from utils import get_args

from visualize import visualize

if __name__ == '__main__':
    args = get_args()
    args.images_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"
    args.mats_dir = "/home/robert/PycharmProjects/3DFaceReconstruction/300W-3D/ALL_DATA"
    args.checkpoint = "/home/robert/PycharmProjects/3D-face-reconstruction/checkpoints/simple_model_160"
    data = dataloaders.FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    )


    model = UNet(3, 1)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"Loaded model {args.checkpoint}")

    i = np.random.randint(len(data))
    img, _ = data[i]

    img = to_cuda(img.unsqueeze(0), True)
    #out = F.sigmoid(model(img))
    out = model(img).detach().squeeze(0).squeeze(0).cpu().numpy()
    print(out.shape, type(out))

    mat = np.zeros((192, 192, 200), dtype=np.uint8)
    for i in range(192):
       for j in range(192):
            mat[i, j, :int(out[i, j])] = 1

    img = img.squeeze(0).cpu()
    #out = out.detach().squeeze(0).cpu()

    #visualize(img, out)
    visualize(img, torch.from_numpy(mat.transpose((2, 0, 1))))
