import torch
import dataloader
import datatransform
import torch.optim as optim
import torchvision.transforms as transforms


# data loader
transform = transforms.Compose([
    datatransform.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = dataloader.FacesWith3DCoords(images_dir="300W-3D/AFW", mats_dir="300W-3D/AFW", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model #TODO
net = None

# criterion & optimiezer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def train():
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            imgs2D, imgs3D = data
            imgs2D = num_flat_features(imgs2D)
            imgs3D = num_flat_features(imgs3D)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out_imgs3D = net(imgs2D)
            loss = criterion(out_imgs3D, imgs3D)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == "__main__":
    train()
