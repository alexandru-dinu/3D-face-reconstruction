import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


import dataloaders
from u_net import UNet, to_cuda
from hourglass import StackedHourGlass
from utils import get_args

args = get_args()

trainset = dataloaders.FacesWith3DCoords(
    images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=True, num_workers=8
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
net = StackedHourGlass(nChannels=224, nStack=2, nModules=2, numReductions=4, nOutputs=200)

net.cuda()

# criterion & optimiezer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.RMSprop(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def train():

    if args.resume:
        net.load_state_dict(torch.load(args.checkpoint))

    for epoch in range(1, 1000):  # loop over the dataset multiple times
        print("=== Epoch", epoch, "===")
        scheduler.step()

        running_loss = 0.0
        epoch_avg = 0.0
        run_avg, t = 0.0, 0

        for i, data in tqdm(enumerate(trainloader, start=1)):
            # get the inputs
            imgs2D, imgs3D, landmarks = data


            imgs2D = to_cuda(imgs2D, True)
            imgs3D = to_cuda(imgs3D, True)
            #landmarks = to_cuda(landmarks, True)

            #l_shape = landmarks.shape
            #m = landmarks.view(l_shape[0], l_shape[1], l_shape[2] * l_shape[3]).argmax(2)
            #idx_gt = to_cuda(torch.stack((m // 128, m % 128), dim=2), True) / 128.0

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #out_imgs3D, landmarks_predictions = net(imgs2D, landmarks)

            out_imgs3D = F.sigmoid(net(imgs2D))

            loss = F.binary_cross_entropy(out_imgs3D, imgs3D)
            #loss2 = F.mse_loss(landmarks_predictions, idx_gt)

            #loss = loss1 + 0.5 * loss2
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_avg += loss.item()

            if i % 5 == 0:
                print('[%2d, %5d/%5d] loss: %.8f lr %.8f' % (epoch, i, len(trainloader), running_loss / 5, scheduler.get_lr()[0]))
                running_loss = 0.0

        print("EPOCH AVG", epoch_avg / len(trainloader))
        epoch_avg = 0.0

        if epoch % 5 == 0:
            torch.save(net.state_dict(), "../checkpoints_hourglass/2hourglass_%d_schd_with_aug" % epoch)


if __name__ == "__main__":
    train()
