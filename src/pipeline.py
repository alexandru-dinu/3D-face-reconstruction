import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
import dataloaders
from u_net import UNet, to_cuda
from utils import get_args

args = get_args()

trainset = dataloaders.FacesWith3DCoords(
    images_dir=args.images_dir, mats_dir=args.mats_dir, lands_dir=args.lands_dir, transform=args.transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
#net = UNet(71, 1)
#net = torch.nn.Sequential(
#    UNet(71, 64),
#    UNet(64, 64),
#    UNet(64, 64),
#    UNet(64, 1),
#)


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
net.cuda()

# criterion & optimiezer
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


optimizer = optim.Adam(net.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def train():
    running_avg = 0.0

    for epoch in range(1, 410):  # loop over the dataset multiple times
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
            #out_imgs3D = F.sigmoid(net(imgs2D))
            out_imgs3D = net(imgs2D)
            print(out_imgs3D.shape)
            out_imgs3D = out_imgs3D.reshape(-1, num_flat_features(out_imgs3D))

            #loss = -torch.mean(imgs3D * torch.log(out_imgs3D) + (1 - imgs3D) * torch.log(1 - out_imgs3D))
            loss = criterion(imgs3D, out_imgs3D) + 100 * torch.sum((1 - (imgs3D > 1e-12).type(torch.FloatTensor)).cuda() * out_imgs3D)

            running_avg = loss if i == 1 else running_avg * i / (i + 1) + loss / (i + 1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:
                print('[%2d, %5d] loss: %.8f, run_avg: %.8f' % (epoch, i, running_loss / 1, running_avg))
                running_loss = 0.0

        if epoch % 10 == 0:
            torch.save(net.state_dict(), "../checkpoints/simple_model_%d" % epoch)


if __name__ == "__main__":
    train()
