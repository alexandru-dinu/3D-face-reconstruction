import numpy as np
import torch
import torch.nn.functional as F

import dataloaders
from u_net import UNet, to_cuda
from utils import get_args

from visualize import visualize

if __name__ == '__main__':
    args = get_args()
    data = dataloaders.FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    )

    model = UNet(3, 0)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"Loaded model {args.checkpoint}")

    i = np.random.randint(len(data))
    img, _ = data[i]

    img = to_cuda(img.unsqueeze(0), True)
    out = F.sigmoid(model(img))

    img = img.squeeze(0).cpu()
    out = out.detach().squeeze(0).cpu()

    visualize(img, out)
