import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class VGGModel(nn.Module):
    def __init__(self, out_size):
        super(VGGModel, self).__init__()
        self.feature = torchvision.models.vgg19(pretrained=True).features

        for p in self.feature.parameters():
            p.requires_grad = False

        self.gen = nn.Sequential(
            nn.Linear(25088, out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(out_size, out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4)
        )


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.gen(x)

        return F.relu(x)
