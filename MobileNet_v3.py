import torch
import torch.nn as nn
import config
from importlib import reload
config = reload(config)


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

    def forward(self, inputs):
        out = []
        for channel, layer in enumerate(self.depth_wise):
            out.append(layer(inputs[:, [channel]]))
        return self.non_linear(self.normalization(torch.cat(out, dim=1)))


class DepthWiseSepConv(nn.Module):
    def __init__(self, in_c, out_c, s, k, nl, se):
        super().__init__()
        self.se = se
        # add squeeze and excite
        if self.se:
            self.sae = SqueezeAndExcide(in_c, out_c)
        self.depth_wise_conv = DepthWiseConv(in_c, s, k, nl)
        self.point_wise = nn.Conv2d(in_c, out_c, 1)
        self.normalization = nn.BatchNorm2d(out_c)

    def forward(self, inputs):
        dw_out = self.depth_wise_conv(inputs)
        out = self.point_wise(dw_out)
        if self.se:
            out *= self.sae(dw_out)

        return self.normalization(out)


class BottleNeck(BaseLayer):
    def __init__(self, in_c, exp_c, out_c, k, se, nl, s):
        super().__init__()
        self.non_linear = self.choice_nl(nl)
        self.conv = nn.Conv2d(in_c, exp_c, 1)
        self.depth_wise_sep = DepthWiseSepConv(exp_c, out_c, s, k, nl, se)
        self.normalization = nn.BatchNorm2d(exp_c)

    def forward(self, inputs):
        out = self.non_linear(self.normalization(self.conv(inputs)))
        out = self.depth_wise_sep(out)

        return out + inputs if inputs.size == out.size else out


class Conv(BaseLayer):
    def __init__(self, in_c, out_c, k, bn, se, nl, s):
        super().__init__()
        self.se = se
        self.bn = bn

        if self.se:
            self.sae = SqueezeAndExcide(in_c, out_c)
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


def get_model():
    model = nn.Sequential()
    for ind, param in enumerate(config.LARGE_PARAMS):
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
