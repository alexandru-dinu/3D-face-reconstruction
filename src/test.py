import cv2
import torch
import torch.nn.functional as F

from model_hourglass import StackedHourGlass
from utils import get_args, to_cuda
from visualize import visualize

from out_to_mesh import get_coords, to_stl

if __name__ == '__main__':
    args = get_args()

    # data = dataloaders.FacesWith3DCoords(
    #    images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
    # )

    img = cv2.imread(args.test_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    img = (img - 128) / 128.0

    model = StackedHourGlass(nChannels=224, nStack=2, nModules=2, numReductions=4, nOutputs=200)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"[+] Loaded model {args.checkpoint}")

    # i = np.random.randint(len(data))
    # img, _ = data[0]

    img = to_cuda(img.unsqueeze(0), True)

    out = F.sigmoid(model(img))

    if False:
        out, _ = model(img, landmarks.unsqueeze(0))
        out = F.sigmoid(out)

    img = img.squeeze(0).cpu()
    out = out.detach().squeeze(0).cpu()

    # view --

    thr, sz = 0.9, 0.25

    x, y, z = get_coords(out.numpy(), sz=sz, thr=thr)
    to_stl("../meshes/out_test", x, y, z)

    visualize(img, out, sz=0.25, thr=thr)
