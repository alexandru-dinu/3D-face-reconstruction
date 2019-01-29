import torch
import torch.nn.functional as F
import torch.optim as optim

import dataloaders
from u_net import UNet, to_cuda
from hourglass import StackedHourGlass
from utils import get_args

args = get_args()

trainset = dataloaders.FacesWith3DCoords(
    images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
net = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nOutputs=200)
net.cuda()

# criterion & optimiezer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def train():
    running_avg = 0.0

    for epoch in range(1, 1000):  # loop over the dataset multiple times
        print("=== Epoch", epoch, "===")
        running_loss = 0.0

        for i, data in enumerate(trainloader, start=1):
            # get the inputs
            imgs2D, imgs3D = data
            imgs3D = imgs3D.reshape(-1, num_flat_features(imgs3D))

            imgs2D = to_cuda(imgs2D, True)
            imgs3D = to_cuda(imgs3D, True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out_imgs3D = F.sigmoid(net(imgs2D))

            out_imgs3D = out_imgs3D.reshape(-1, num_flat_features(out_imgs3D))

            loss = -torch.mean(imgs3D * torch.log(out_imgs3D) + (1 - imgs3D) * torch.log(1 - out_imgs3D))
            # loss = torch.nn.MSELoss()(imgs3D, out_imgs3D)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()

        print('[%2d, %5d] loss: %.8f' % (epoch, i, running_loss / i))

        if epoch % 50 == 0:
            torch.save(net.state_dict(), "../checkpoints/simple_model_%d_H1" % epoch)
        if epoch == 180:
            import pdb; pdb.set_trace()


if __name__ == "__main__":
    train()
