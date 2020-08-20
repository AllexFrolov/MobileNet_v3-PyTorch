import torch
import torch.nn as nn
import config
from importlib import reload
config = reload(config)


def compute_out_size(in_size, stride, padding=1, kernel_size=3,):
    return int((in_size + 2 * padding - kernel_size) / stride + 1)


class DepthWiseSepConv(nn.Module):
    def __init__(self, input_ch, output_ch, stride):
        super().__init__()
        self.depth_wise_convs = nn.ModuleList(
            [nn.Conv2d(1, 1, 3, stride, padding=1) for _ in range(input_ch)]
        )
        self.point_wise = nn.Conv2d(input_ch, output_ch, 1)
        self.batch_norm_1 = nn.BatchNorm2d(input_ch)
        self.batch_norm_2 = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        input_channels = inputs.size()[1]
        result = []
        for channel in range(input_channels):
            result.append(self.depth_wise_convs[channel](inputs[:, [channel]]))
        out = torch.cat(result, dim=1)
        out = self.relu(self.batch_norm_1(out))
        return self.relu(self.batch_norm_2(self.point_wise(out)))


class Head(nn.Module):
    def __init__(self, im_size, in_channels=1,
                 out_channels=32, kernel_size=3, stride=2):
        super().__init__()
        self.im_size_ = \
            compute_out_size(im_size, stride, kernel_size=kernel_size)
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        out = self.model(inputs)
        return out

    def get_out_size(self):
        return self.im_size_, self.out_channels


class Blocks(nn.Module):
    def __init__(self, im_size, params, alpha=1):
        super().__init__()
        self.im_size_ = im_size
        self.out_channels = int(params[-1].out_channels*alpha)
        blocks = []
        for param in params:
            self.im_size_ = \
                compute_out_size(self.im_size_, param.stride)
            blocks.append(DepthWiseSepConv(int(param.in_channels*alpha),
                                           int(param.out_channels*alpha),
                                           param.stride))

        self.model = nn.Sequential(*blocks,
                                   nn.AvgPool2d(self.im_size_)
                                   )
        self.im_size_ = \
            compute_out_size(self.im_size_, self.im_size_, 0, self.im_size_)

    def forward(self, inputs):
        return self.model(inputs)

    def get_out_size(self):
        return self.im_size_, self.out_channels


class MobileNet(nn.Module):
    def __init__(self, head, blocks, n_classes):
        super().__init__()
        self.head = head
        self.blocks = blocks
        im_size, channels = self.blocks.get_out_size()

        self.floor = nn.Sequential(
            nn.Flatten(),
            nn.Linear((im_size**2)*channels, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs):
        out = self.head(inputs)
        out = self.blocks(out)
        return self.floor(out)


def get_model(alpha=1, ro=1):
    if alpha <= 0:
        raise ValueError('alpha should be in range (0, 1]')
    head = Head(int(config.IM_SIZE*ro), config.IN_CHANNELS, int(config.OUT_CHANNELS*alpha),
                stride=config.STRIDE)
    blocks = Blocks(head.get_out_size()[0], config.PARAMS, alpha)

    return MobileNet(head, blocks, config.CLASSES)
