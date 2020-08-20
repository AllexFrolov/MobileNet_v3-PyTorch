import torch
import torch.nn as nn
import config
from importlib import reload
config = reload(config)


def compute_out_size(in_size, stride, padding, kernel_size=3):
    return int((in_size + 2 * padding - kernel_size) / stride + 1)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.depthwise = \
            nn.ModuleList(
                [nn.Conv2d(1, 1, 3, stride, padding=1) for _ in range(in_channels)]
            )
        self.non_linear = nn.ReLU6()
        self.normalization = nn.BatchNorm2d(in_channels)

    def forward(self, inputs):
        out = []
        for channel, layer in enumerate(self.depthwise):
            out.append(layer(inputs[:, [channel]]))
        return self.non_linear(self.normalization(torch.cat(out, dim=1)))


class DepthWiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseSepConv, self).__init__()
        self.depth_wise_conv = DepthWiseConv(in_channels, stride)
        self.point_wise = nn.Conv2d(in_channels, out_channels, 1)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        out = self.depth_wise_conv(inputs)
        out = self.point_wise(out)
        return self.normalization(out)


class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, t, s):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * t, 1)
        self.depth_wise_sep = DepthWiseSepConv(in_c * t, out_c, s)
        self.normalization = nn.BatchNorm2d(in_c * t)
        self.non_linear = nn.ReLU6()
        self.stride = s

    def forward(self, inputs):
        out = self.non_linear(self.normalization(self.conv(inputs)))
        out = self.depth_wise_sep(out)

        return out if inputs.size != out.size else out + inputs


def get_model():
    model = nn.Sequential()
    im_size = config.IM_SIZE
    for ind, param in enumerate(config.PARAMS_V2):
        param_name = type(param).__name__
        if param_name == 'Conv':
            model.add_module(
                f'{ind} {param_name}',
                nn.Sequential(
                    nn.Conv2d(param.in_c, param.c, param.k, param.s, param.p),
                    nn.BatchNorm2d(param.c),
                    nn.ReLU6()
                )
            )
            im_size = compute_out_size(im_size, param.s, param.p, param.k)
        elif param_name == 'BottleNeck':
            model.add_module(
                f'{ind} {param_name}',
                nn.Sequential(
                    BottleNeck(param.in_c, param.c, param.t, param.s),
                    *[BottleNeck(param.c, param.c, param.t, 1) for _ in range(param.n - 1)]
                )
            )
            im_size = compute_out_size(im_size, param.s, 1)

        elif param_name == 'Floor':
            model.add_module(
                param_name,
                nn.Sequential(
                    nn.AvgPool2d(im_size),
                    nn.Conv2d(param.in_c, param.c, param.k, param.s),
                    nn.Flatten(),
                    nn.LogSoftmax(dim=0)
                )
            )
            im_size = compute_out_size(im_size, 7, 0, 7)

    return model