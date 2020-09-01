import math
from collections import namedtuple

import torch.nn as nn
from torch import Tensor


class CorrectDepth:
    def __init__(self, alpha: float, min_depth: int = 3):
        self.alpha = alpha
        self.min_depth = min_depth

    def __call__(self, depth: int) -> int:
        """
            Adjust a depth value
            :param depth: (int) Depth/channels
            :return: (int) corrected depth
        """
        return max(self.min_depth, int(depth * self.alpha))


def get_model_params(architecture: str, classifier_output: int, c_d: CorrectDepth) -> tuple:
    """
    Return corrected architecture parameters
    :param architecture: (str) Architecture should be "large" or "small"
    :param classifier_output: output from classifier layer
    :param c_d: (CorrectDepth) Depth corrector
    :return: (tuple of namedtuple) corrected architecture parameters
    """
    BNeck = namedtuple('bneck', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))
    Conv = namedtuple('conv', ('in_c', 'out_c', 'k', 'bn', 'se', 'nl', 's'))
    Pool = namedtuple('pool', ('in_c', 'exp_c', 'out_c', 'k', 'se', 'nl', 's'))

    if architecture == 'large':
        return (
            Conv(3, c_d(16), 3, True, False, 'HS', 2),  # 224
            BNeck(c_d(16), c_d(16), c_d(16), 3, False, 'RE', 1),  # 112
            BNeck(c_d(16), c_d(64), c_d(24), 3, False, 'RE', 2),  # 112
            BNeck(c_d(24), c_d(72), c_d(24), 3, False, 'RE', 1),  # 56
            BNeck(c_d(24), c_d(72), c_d(40), 5, True, 'RE', 2),  # 56
            BNeck(c_d(40), c_d(120), c_d(40), 5, True, 'RE', 1),  # 28
            BNeck(c_d(40), c_d(120), c_d(40), 5, True, 'RE', 1),  # 28
            BNeck(c_d(40), c_d(240), c_d(80), 3, False, 'HS', 2),  # 28
            BNeck(c_d(80), c_d(200), c_d(80), 3, False, 'HS', 1),  # 14
            BNeck(c_d(80), c_d(184), c_d(80), 3, False, 'HS', 1),  # 14
            BNeck(c_d(80), c_d(184), c_d(80), 3, False, 'HS', 1),  # 14
            BNeck(c_d(80), c_d(480), c_d(112), 3, True, 'HS', 1),  # 14
            BNeck(c_d(112), c_d(672), c_d(112), 3, True, 'HS', 1),  # 14
            BNeck(c_d(112), c_d(672), c_d(160), 5, True, 'HS', 2),  # 14
            BNeck(c_d(160), c_d(960), c_d(160), 5, True, 'HS', 1),  # 7
            BNeck(c_d(160), c_d(960), c_d(160), 5, True, 'HS', 1),  # 7
            Conv(c_d(160), c_d(960), 1, True, False, 'HS', 1),  # 7
            Pool(c_d(960), '-', '-', '-', False, None, 1),  # 7
            Conv(c_d(960), c_d(1280), 1, False, False, 'HS', 1),  # 1
            Conv(c_d(1280), classifier_output, 1, False, False, None, 1),  # 1
        )
    elif architecture == 'small':
        return (
            Conv(3, c_d(16), 3, True, False, 'HS', 2),  # 224
            BNeck(c_d(16), c_d(16), c_d(16), 3, True, 'RE', 2),  # 112
            BNeck(c_d(16), c_d(72), c_d(24), 3, False, 'RE', 2),  # 56
            BNeck(c_d(24), c_d(88), c_d(24), 3, False, 'RE', 1),  # 28
            BNeck(c_d(24), c_d(96), c_d(40), 5, True, 'HS', 2),  # 28
            BNeck(c_d(40), c_d(240), c_d(40), 5, True, 'HS', 1),  # 14
            BNeck(c_d(40), c_d(240), c_d(40), 5, True, 'HS', 1),  # 14
            BNeck(c_d(40), c_d(120), c_d(48), 5, True, 'HS', 1),  # 14
            BNeck(c_d(48), c_d(144), c_d(48), 5, True, 'HS', 1),  # 14
            BNeck(c_d(48), c_d(288), c_d(96), 5, True, 'HS', 2),  # 14
            BNeck(c_d(96), c_d(96), c_d(96), 5, True, 'HS', 1),  # 7
            BNeck(c_d(96), c_d(576), c_d(96), 5, True, 'HS', 1),  # 7
            Conv(c_d(96), c_d(576), 1, True, True, 'HS', 1),  # 7
            Pool(c_d(576), '-', '-', '-', False, None, 1),  # 7
            Conv(c_d(576), c_d(1024), 1, True, False, 'HS', 1),  # 1
            Conv(c_d(1024), classifier_output, 1, True, False, None, 1),  # 1
        )
    else:
        raise ValueError('size must be "large" or "small"')


class BaseLayer(nn.Module):
    @staticmethod
    def choice_nl(nl):
        if nl == 'RE':
            return nn.ReLU()
        elif nl == 'HS':
            return nn.Hardswish()
        elif nl is None:
            return None
        else:
            raise ValueError('nl should be "RE", "HS" or None')

    @staticmethod
    def same_padding(kernel_size):
        return (kernel_size - 1) // 2


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        sqz_channels = math.ceil(channels / 4)
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, sqz_channels, 1),
            nn.ReLU(),
            nn.Conv2d(sqz_channels, channels, 1),
            nn.Hardsigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.sequential(inputs)


class DepthWiseConv(BaseLayer):
    def __init__(self, channels: int, kernel_size: int or tuple,
                 non_linear: str, stride: int = 1):
        super().__init__()

        self.depth_wise = nn.Conv2d(channels,
                                    channels,
                                    kernel_size,
                                    stride,
                                    self.same_padding(kernel_size),
                                    groups=channels)

        self.non_linear = self.choice_nl(non_linear)
        self.normalization = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.depth_wise(inputs)
        return self.non_linear(self.normalization(out))


class DepthWiseSepConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 squeeze_excite_add: bool, non_linear: str, stride: int = 1):
        super().__init__()
        # add Squeeze and Excitation block
        if squeeze_excite_add:
            self.sae = SqueezeAndExcite(in_channels)
        else:
            self.sae = None

        self.depth_wise_conv = DepthWiseConv(in_channels, kernel_size, non_linear, stride)
        self.point_wise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.depth_wise_conv(inputs)
        if self.sae is not None:
            out = self.sae(out)
        out = self.point_wise(out)
        return out


class BottleNeck(BaseLayer):
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 kernel_size: int or tuple, squeeze_excite_add: bool,
                 non_linear: str, stride: int = 1):
        super().__init__()
        self.non_linear = self.choice_nl(non_linear)
        self.expansion_layer = nn.Conv2d(in_channels, exp_channels, 1)
        self.depth_wise_sep = \
            DepthWiseSepConv(exp_channels, out_channels, kernel_size,
                             squeeze_excite_add, non_linear, stride)

        self.normalization_bn = nn.BatchNorm2d(exp_channels)
        self.normalization_out = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.expansion_layer(inputs)
        out = self.normalization_bn(out)
        out = self.non_linear(out)
        out = self.depth_wise_sep(out)
        out = out + inputs if inputs.size() == out.size() else out
        return self.normalization_out(out)


class Convolution(BaseLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 batch_norm_add: bool, squeeze_excite_add: bool, non_linear: str, stride: int = 1):
        super().__init__()
        # add Squeeze and Excitation block
        if squeeze_excite_add:
            self.sae = SqueezeAndExcite(out_channels)
        else:
            self.sae = None
        # add batch normalization
        if batch_norm_add:
            self.normalization = nn.BatchNorm2d(out_channels)
        else:
            self.normalization = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, self.same_padding(kernel_size))
        self.non_linear = self.choice_nl(non_linear)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        if self.sae is not None:
            out = self.sae(out)
        if self.normalization is not None:
            out = self.normalization(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out


def weight_initialization(model):
    """
    Initialization model weight
    :param model: (nn.Module) model
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def get_model(classes, size='small', alpha=1., dropout=0.8, min_depth=3):
    correct_depth = CorrectDepth(alpha, min_depth)
    parameters = get_model_params(size, classes, correct_depth)
    model = nn.Sequential()
    for ind, param in enumerate(parameters[:-1]):
        layer_name = type(param).__name__
        if layer_name == 'conv':
            model.add_module(f'{ind} {layer_name}', Convolution(*param))
        elif layer_name == 'bneck':
            model.add_module(f'{ind} {layer_name}', BottleNeck(*param))
        elif layer_name == 'pool':
            model.add_module(layer_name, nn.AdaptiveAvgPool2d(1))

    model.add_module('Dropout', nn.Dropout(dropout))
    model.add_module('Classifier', Convolution(*parameters[-1]))
    model.add_module('Flatten', nn.Flatten())
    weight_initialization(model)
    return model
