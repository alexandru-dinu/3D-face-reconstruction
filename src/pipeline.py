import torch
import torch.nn.functional as F
import dataloaders
import datatransform
import torch.optim as optim
import torchvision.transforms as transforms
from u_net import UNet, to_cuda

trainset = dataloaders.FacesWith3DCoords(images_dir="300W-3D/ALL_DATA", mats_dir="300W-3D/ALL_DATA", transform=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
net = UNet(3, 0)
net.cuda()

# criterion & optimiezer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def train():
    for epoch in range(10):  # loop over the dataset multiple times
        print("=== Epoch", epoch, "===")
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
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

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0


        torch.save(net, "simple_model")

if __name__ == "__main__":
    train()
