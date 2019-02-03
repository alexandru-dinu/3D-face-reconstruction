import torch
import torch.nn.functional as F
import torch.optim as optim

import data_loader
from model_hourglass import StackedHourGlass
from utils import get_args, to_cuda

args = get_args()

trainset = data_loader.FacesWith3DCoords(
    images_dir=args.images_dir, mats_dir=args.mats_dir, transform=args.transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=2, shuffle=True, num_workers=2
)

model = StackedHourGlass(nChannels=224, nStack=2, nModules=2, numReductions=4, nOutputs=200)
model.cuda()
model.train()
print("Set-up model")

# criterion & optimizer
criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def train():
    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint))

    for epoch in range(args.start_epoch, 1000):  # loop over the dataset multiple times
        print("=== Epoch", epoch, "===")
        scheduler.step()

        running_loss, epoch_avg = 0.0, 0.0

        for i, data in enumerate(trainloader, start=1):
            # get the inputs
            images, volumes, landmarks = data

            images = to_cuda(images, True)
            volumes = to_cuda(volumes, True)
            # landmarks = to_cuda(landmarks, True)

            # l_shape = landmarks.shape
            # m = landmarks.view(l_shape[0], l_shape[1], l_shape[2] * l_shape[3]).argmax(2)
            # idx_gt = to_cuda(torch.stack((m // 128, m % 128), dim=2), True) / 128.0

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # out_volumes, landmarks_predictions = net(images, landmarks)

            out_volumes = F.sigmoid(model(images))

            loss = F.binary_cross_entropy(out_volumes, volumes)
            # loss2 = F.mse_loss(landmarks_predictions, idx_gt)
            # loss = loss1 + 0.5 * loss2

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_avg += loss.item()

            if i % 1 == 0:
                print('[%2d, %5d/%5d] loss: %.8f lr %.8f' % (epoch, i, len(trainloader), running_loss / 1, scheduler.get_lr()[0]))
                running_loss = 0.0

        print("EPOCH AVG", epoch_avg / len(trainloader))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "../checkpoints/2hourglass_%d_schd_with_aug" % epoch)


if __name__ == "__main__":
    train()
