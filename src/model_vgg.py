import torch
import torch.nn.functional as F
import torchvision.models


class VGGModel(torch.nn.Module):
    def __init__(self, out_size):
        super(VGGModel, self).__init__()
        self.feature = torchvision.models.vgg19(pretrained=True).features

        for p in self.feature.parameters():
            p.requires_grad = False

        # last conv gives 7x7x512 = 25088
        self.linear1 = torch.nn.Linear(25088, out_size)
        self.linear2 = torch.nn.Linear(out_size, out_size)


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x
