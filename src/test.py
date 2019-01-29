import numpy as np
import torch
import torch.nn.functional as F

import dataloaders
from u_net import UNet, to_cuda
from utils import get_args

from visualize import visualize
import torchvision.models as models




class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.feature = models.vgg19(pretrained=True).features
        for p in self.feature.parameters():
            p.requires_grad=False
        self.linear1 = torch.nn.Linear(25088, 50 * 50)
        self.linear2 = torch.nn.Linear(50 * 50, 50 * 50)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x




net = M()

if __name__ == '__main__':
    args = get_args()
    args.images_dir = "../300W-3D-all/images"
    args.mats_dir = "../300W-3D-all/3d-scans"
    args.lands_dir = "../300W-3D-all/landmarks"
    args.checkpoint = "/home/robert/PycharmProjects/3D-face-reconstruction/checkpoints/simple_model_60"


    data = dataloaders.FacesWith3DCoords(
        images_dir=args.images_dir, mats_dir=args.mats_dir, lands_dir=args.lands_dir, transform=args.transform
    )

    model = M()
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"Loaded model {args.checkpoint}")

    i = np.random.randint(len(data))
    img, _ = data[i]

    img = to_cuda(img.unsqueeze(0), True)
    #out = F.sigmoid(model(img))
    out = model(img).detach().squeeze(0).cpu().numpy().reshape((50, 50))
    print(out.shape, type(out), out.max())

    mat = np.zeros((50, 50, 200), dtype=np.uint8)
    for i in range(50):
       for j in range(50):
            mat[i, j, :int(out[i, j])] = 1

    img = img.squeeze(0).cpu()
    #out = out.detach().squeeze(0).cpu()
    import datatransform
    import cv2
    mat = datatransform.Resize()(mat, 200)
    mat = cv2.GaussianBlur(mat, (5, 5), sigmaX=7, sigmaY=7)

    #visualize(img, out)
    visualize(img[:3, :, :], torch.from_numpy(mat.transpose((2, 0, 1))))
