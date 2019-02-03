import numpy as np
import torch
import torch.nn.functional as F

import data_loader
import visualize
from utils import get_args, to_cuda

nn = torch.load("simple_model2").cuda()
# img = cv2.imread("test2.jpg", cv2.IMREAD_COLOR)
# img = cv2.resize(img, (200, 200))
# img = torch.from_numpy(im.transpose(2, 0, 1))

args = get_args()
trainset = data_loader.FacesWith3DCoords(
    images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
)

i = np.random.randint(len(trainset))
img, vol = trainset[1]

img = img.unsqueeze(0)
img = to_cuda(img, True)

with torch.no_grad():
    img3d = F.sigmoid(nn(img))
    print(img3d[-1].shape)
    visualize.visualize(img[0].cpu(), img3d[0].cpu())
    visualize.visualize(img[0].cpu(), vol.cpu())
