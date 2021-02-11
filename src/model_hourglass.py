import torch.nn.functional as F

from model_hourglass_parts import *


class Hourglass(nn.Module):
    """docstring for Hourglass"""

    def __init__(
        self,
        nChannels=256,
        numReductions=4,
        nModules=2,
        poolKernel=(3, 3),
        poolStride=(3, 3),
        upSampleKernel=3,
    ):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if numReductions > 1:
            self.hg = Hourglass(
                self.nChannels,
                self.numReductions - 1,
                self.nModules,
                self.poolKernel,
                self.poolStride,
            )
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(Residual(self.nChannels, self.nChannels))

            self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(Residual(self.nChannels, self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        Upsampling Layer (Can we change this??????)
        As per Newell's paper upsamping recommended
        """
        self.up = nn.Upsample(scale_factor=upSampleKernel, mode="nearest")

    def forward(self, x):

        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions > 1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)

        out2 = self.up(out2)

        diffY = out1.size()[2] - out2.size()[2]
        diffX = out1.size()[3] - out2.size()[3]

        out2 = F.pad(
            out2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        )

        return out2 + out1


class StackedHourGlass(nn.Module):
    """docstring for StackedHourGlass"""

    def __init__(
        self,
        nChannels,
        nStack,
        nModules,
        numReductions,
        nInputs1=3,
        nInputs2=68,
        nOutputs=2,
    ):
        super(StackedHourGlass, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.nOutputs = nOutputs

        self.res1 = Residual(nInputs1, 64)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = Residual(64, 128)
        self.res3 = Residual(128, self.nChannels)

        # FIXME - TEMPORARY!!!
        self.__with_lands = False
        self.__with_aug = True

        # FIXME
        if self.__with_lands or self.__with_aug:
            self.res_lands = Residual(nInputs2, self.nChannels)

        _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(self.nStack):
            _hourglass.append(
                Hourglass(self.nChannels, self.numReductions, self.nModules)
            )
            _ResidualModules = []
            for _ in range(self.nModules):
                _ResidualModules.append(Residual(self.nChannels, self.nChannels))
            _ResidualModules = nn.Sequential(*_ResidualModules)
            _Residual.append(_ResidualModules)
            _lin1.append(BnReluConv(self.nChannels, self.nChannels))
            _chantojoints.append(nn.Conv2d(self.nChannels, self.nOutputs, 1))
            _lin2.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
            _jointstochan.append(nn.Conv2d(self.nOutputs, self.nChannels, 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin1 = nn.ModuleList(_lin1)
        self.chantojoints = nn.ModuleList(_chantojoints)
        self.lin2 = nn.ModuleList(_lin2)
        self.jointstochan = nn.ModuleList(_jointstochan)

        # FIXME
        if self.__with_lands or self.__with_aug:
            self.lands_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.nChannels, out_channels=68, kernel_size=3, stride=1
                ),
                nn.ReLU(inplace=True),
            )
            self.lands_lin = nn.Linear(15876, 2)

    def forward(self, imgs, landmarks_maps=None):

        x = self.res1(imgs)
        x = self.res2(x)
        x = self.res3(x)

        # FIXME
        if self.__with_lands:
            landmarks_maps = self.res_lands(landmarks_maps)

        for i in range(self.nStack):
            x1 = self.hourglass[i](x)
            x1 = self.Residual[i](x1)
            x1 = self.lin1[i](x1)
            out = self.chantojoints[i](x1)
            x1 = self.lin2[i](x1)

            # FIXME
            if self.__with_lands:
                if i == 0:
                    out_inter = self.lands_conv(x1)
                    out_inter = out_inter.view(x1.shape[0], 68, 126 * 126)
                    out_inter = self.lands_lin(out_inter)

            # FIXME
            if self.__with_lands:
                x = x + x1 + landmarks_maps + self.jointstochan[i](out)

            # FIXME
            if not self.__with_lands:
                x = x + x1 + self.jointstochan[i](out)

        # FIXME
        if not self.__with_lands:
            return out

        # FIXME
        if self.__with_lands:
            return out, out_inter
