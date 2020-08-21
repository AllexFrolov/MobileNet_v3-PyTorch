import torch
import torch.nn as nn
import config
from collections import namedtuple
from importlib import reload
config = reload(config)

BNeck = namedtuple('bneck', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))
Conv = namedtuple('conv', ('in_c', 'out_c', 'k', 'bn', 'se', 'nl', 's'))
Pool = namedtuple('pool', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))
LARGE_PARAMS = (
    Conv(1, 16, 3, True, False, 'HS', 2),   # 224
    BNeck(16, 16, 16, 3, False, 'RE', 1),  # 112
    BNeck(16, 64, 24, 3, False, 'RE', 2),  # 112
    BNeck(24, 72, 24, 3, False, 'RE', 1),  # 56
    BNeck(24, 72, 40, 5, True, 'RE', 2),  # 56
    BNeck(40, 120, 40, 5, True, 'RE', 1),  # 28
    BNeck(40, 120, 40, 5, True, 'RE', 1),  # 28
    BNeck(40, 240, 80, 3, False, 'HS', 2),  # 28
    BNeck(80, 200, 80, 3, False, 'HS', 1),  # 14
    BNeck(80, 184, 80, 3, False, 'HS', 1),  # 14
    BNeck(80, 184, 80, 3, False, 'HS', 1),  # 14
    BNeck(80, 480, 112, 3, True, 'HS', 1),  # 14
    BNeck(112, 672, 112, 3, True, 'HS', 1),  # 14
    BNeck(112, 672, 160, 5, True, 'HS', 2),  # 14
    BNeck(160, 960, 160, 5, True, 'HS', 1),  # 7
    BNeck(160, 960, 160, 5, True, 'HS', 1),  # 7
    Conv(160, 960, 1, True, False, 'HS', 1),  # 7
    Pool(960, '-', '-', '-', False, '-', 1),  # 7
    Conv(960, 1280, 1, False, False, 'HS', 1),  # 1
    Conv(1280, config.CLASSES, 1, False, False, '-', 1),  # 1
)

SMALL_PARAMS = (
    Conv(1, 16, 3, True, False, 'HS', 2),   # 224
    BNeck(16, 16, 16, 3, True, 'RE', 2),  # 112
    BNeck(16, 72, 24, 3, False, 'RE', 2),  # 56
    BNeck(24, 88, 24, 3, False, 'RE', 1),  # 28
    BNeck(24, 96, 40, 5, True, 'HS', 2),  # 28
    BNeck(40, 240, 40, 5, True, 'HS', 1),  # 14
    BNeck(40, 240, 40, 5, True, 'HS', 1),  # 14
    BNeck(40, 120, 48, 5, True, 'HS', 2),  # 14
    BNeck(48, 144, 48, 5, True, 'HS', 1),  # 14
    BNeck(48, 288, 96, 5, True, 'HS', 1),  # 14
    BNeck(96, 576, 96, 5, True, 'HS', 1),  # 7
    BNeck(96, 576, 96, 5, True, 'HS', 1),  # 7
    Conv(96, 576, 1, True, True, 'HS', 1),  # 7
    Pool(576, '-', '-', '-', False, '-', 1),  # 7
    Conv(576, 1024, 1, True, False, 'HS', 1),  # 1
    Conv(1024, config.CLASSES, 1, True, False, '-', 1),  # 1
)


class BaseLayer(nn.Module):
    @staticmethod
    def choice_nl(nl):
        if nl == 'RE':
            return nn.ReLU()
        elif nl == 'HS':
            return nn.Hardswish()
        else:
            return None

    @staticmethod
    def same_padding(kernel_size):
        return (kernel_size - 1) // 2


class SqueezeAndExcide(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_c, int(in_c / 4), 1),
                nn.ReLU(),
                nn.Conv2d(int(in_c / 4), out_c, 1),
                nn.Hardswish()
            )

    def forward(self, inputs):
        return self.model(inputs)


class DepthWiseConv(BaseLayer):
    def __init__(self, channels, stride, k_size, nl):
        super().__init__()

        self.depth_wise = \
            nn.ModuleList(
                [nn.Conv2d(1, 1, k_size, stride, self.same_padding(k_size))
                 for _ in range(channels)]
            )

        self.non_linear = self.choice_nl(nl)
        self.normalization = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, inputs):
        out = []
        for channel, layer in enumerate(self.depth_wise):
            out.append(self.dropout(layer(inputs[:, [channel]])))
        return self.non_linear(self.normalization(torch.cat(out, dim=1)))


class DepthWiseSepConv(nn.Module):
    def __init__(self, in_c, out_c, s, k, nl, se):
        super().__init__()
        self.se = se
        # add squeeze and excite
        if self.se:
            self.sae = SqueezeAndExcide(in_c, in_c)
        self.depth_wise_conv = DepthWiseConv(in_c, s, k, nl)
        self.point_wise = nn.Conv2d(in_c, out_c, 1)

    def forward(self, inputs):
        out = self.depth_wise_conv(inputs)
        if self.se:
            out *= self.sae(out)
        out = self.point_wise(out)

        return out


class BottleNeck(BaseLayer):
    def __init__(self, in_c, exp_c, out_c, k, se, nl, s):
        super().__init__()
        self.non_linear = self.choice_nl(nl)
        self.conv = nn.Conv2d(in_c, exp_c, 1)
        self.depth_wise_sep = DepthWiseSepConv(exp_c, out_c, s, k, nl, se)
        self.normalization_bn = nn.BatchNorm2d(exp_c)
        self.normalization_out = nn.BatchNorm2d(out_c)

    def forward(self, inputs):
        out = self.non_linear(self.normalization_bn(self.conv(inputs)))
        out = self.depth_wise_sep(out)
        out = out + inputs if inputs.size == out.size else out
        return self.normalization_out(out)


class Conv(BaseLayer):
    def __init__(self, in_c, out_c, k, bn, se, nl, s):
        super().__init__()
        self.se = se
        self.bn = bn

        if self.se:
            self.sae = SqueezeAndExcide(out_c, out_c)
        if self.bn:
            self.normalization = nn.BatchNorm2d(out_c)

        self.conv = nn.Conv2d(in_c, out_c, k, s, self.same_padding(k))
        self.non_linear = self.choice_nl(nl)

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.se:
            out *= self.sae(out)
        if self.bn:
            out = self.normalization(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out


def get_model(size='small'):

    if size == 'large':
        parameters = config.LARGE_PARAMS
    else:
        parameters = config.SMALL_PARAMS

    model = nn.Sequential()
    for ind, param in enumerate(parameters):
        layer_name = type(param).__name__
        if layer_name == 'conv':
            model.add_module(f'{ind} {layer_name}', Conv(*param))
        elif layer_name == 'bneck':
            model.add_module(f'{ind} {layer_name}', BottleNeck(*param))
        elif layer_name == 'pool':
            model.add_module(layer_name, nn.AdaptiveAvgPool2d(1))
    model.add_module('LogSoftMax', nn.LogSoftmax(dim=1))
    model.add_module('Flatten', nn.Flatten())
    return model
