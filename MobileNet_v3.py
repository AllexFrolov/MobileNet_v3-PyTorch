import torch
import torch.nn as nn
import config
from collections import namedtuple
from importlib import reload
import math
config = reload(config)

BNeck = namedtuple('bneck', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))
Conv = namedtuple('conv', ('in_c', 'out_c', 'k', 'bn', 'se', 'nl', 's'))
Pool = namedtuple('pool', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))


def correct_depth(depth):
    return max(config.MIN_DEPTH, int(depth * config.ALPHA))


LARGE_PARAMS = (
    Conv(3, correct_depth(16), 3, True, False, 'HS', 2),   # 224
    BNeck(correct_depth(16), correct_depth(16), correct_depth(16), 3, False, 'RE', 1),  # 112
    BNeck(correct_depth(16), correct_depth(64), correct_depth(24), 3, False, 'RE', 2),  # 112
    BNeck(correct_depth(24), correct_depth(72), correct_depth(24), 3, False, 'RE', 1),  # 56
    BNeck(correct_depth(24), correct_depth(72), correct_depth(40), 5, True, 'RE', 2),  # 56
    BNeck(correct_depth(40), correct_depth(120), correct_depth(40), 5, True, 'RE', 1),  # 28
    BNeck(correct_depth(40), correct_depth(120), correct_depth(40), 5, True, 'RE', 1),  # 28
    BNeck(correct_depth(40), correct_depth(240), correct_depth(80), 3, False, 'HS', 2),  # 28
    BNeck(correct_depth(80), correct_depth(200), correct_depth(80), 3, False, 'HS', 1),  # 14
    BNeck(correct_depth(80), correct_depth(184), correct_depth(80), 3, False, 'HS', 1),  # 14
    BNeck(correct_depth(80), correct_depth(184), correct_depth(80), 3, False, 'HS', 1),  # 14
    BNeck(correct_depth(80), correct_depth(480), correct_depth(112), 3, True, 'HS', 1),  # 14
    BNeck(correct_depth(112), correct_depth(672), correct_depth(112), 3, True, 'HS', 1),  # 14
    BNeck(correct_depth(112), correct_depth(672), correct_depth(160), 5, True, 'HS', 2),  # 14
    BNeck(correct_depth(160), correct_depth(960), correct_depth(160), 5, True, 'HS', 1),  # 7
    BNeck(correct_depth(160), correct_depth(960), correct_depth(160), 5, True, 'HS', 1),  # 7
    Conv(correct_depth(160), correct_depth(960), 1, True, False, 'HS', 1),  # 7
    Pool(correct_depth(960), '-', '-', '-', False, '-', 1),  # 7
    Conv(correct_depth(960), correct_depth(1280), 1, False, False, 'HS', 1),  # 1
    Conv(correct_depth(1280), config.CLASSES, 1, False, False, '-', 1),  # 1
)


SMALL_PARAMS = (
    Conv(3, correct_depth(16), 3, True, False, 'HS', 2),   # 224
    BNeck(correct_depth(16), correct_depth(16), correct_depth(16), 3, True, 'RE', 2),  # 112
    BNeck(correct_depth(16), correct_depth(72), correct_depth(24), 3, False, 'RE', 2),  # 56
    BNeck(correct_depth(24), correct_depth(88), correct_depth(24), 3, False, 'RE', 1),  # 28
    BNeck(correct_depth(24), correct_depth(96), correct_depth(40), 5, True, 'HS', 2),  # 28
    BNeck(correct_depth(40), correct_depth(240), correct_depth(40), 5, True, 'HS', 1),  # 14
    BNeck(correct_depth(40), correct_depth(240), correct_depth(40), 5, True, 'HS', 1),  # 14
    BNeck(correct_depth(40), correct_depth(120), correct_depth(48), 5, True, 'HS', 1),  # 14
    BNeck(correct_depth(48), correct_depth(144), correct_depth(48), 5, True, 'HS', 1),  # 14
    BNeck(correct_depth(48), correct_depth(288), correct_depth(96), 5, True, 'HS', 2),  # 14
    BNeck(correct_depth(96), correct_depth(96), correct_depth(96), 5, True, 'HS', 1),  # 7
    BNeck(correct_depth(96), correct_depth(576), correct_depth(96), 5, True, 'HS', 1),  # 7
    Conv(correct_depth(96), correct_depth(576), 1, True, True, 'HS', 1),  # 7
    Pool(correct_depth(576), '-', '-', '-', False, '-', 1),  # 7
    Conv(correct_depth(576), correct_depth(1024), 1, True, False, 'HS', 1),  # 1
    Conv(correct_depth(1024), config.CLASSES, 1, True, False, '-', 1),  # 1
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


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        exp_c = math.ceil(in_c / 4)
        self.model = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_c, exp_c, 1),
                nn.ReLU(),
                nn.Conv2d(exp_c, out_c, 1),
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
        self.normalization = nn.BatchNorm2d(channels, momentum=config.BN_MOMENTUM)
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
            self.sae = SqueezeAndExcite(in_c, in_c)
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
        self.normalization_bn = nn.BatchNorm2d(exp_c, momentum=config.BN_MOMENTUM)
        self.normalization_out = nn.BatchNorm2d(out_c, momentum=config.BN_MOMENTUM)

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
            self.sae = SqueezeAndExcite(out_c, out_c)
        if self.bn:
            self.normalization = nn.BatchNorm2d(out_c, momentum=config.BN_MOMENTUM)

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
        parameters = LARGE_PARAMS
    else:
        parameters = SMALL_PARAMS

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
