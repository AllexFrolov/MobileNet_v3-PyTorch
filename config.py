from collections import namedtuple

# model parameters
IM_SIZE = 224
# head parameters
# IN_CHANNELS = 1
# OUT_CHANNELS = 32
# STRIDE = 2
# # blocks parameters
# DepthSepConv = namedtuple('DepthSeparableConv',
#                           ('in_channels', 'out_channels', 'stride'))
# PARAMS = (DepthSepConv(32, 64, 1),
#           DepthSepConv(64, 128, 2),
#           DepthSepConv(128, 128, 1),
#           DepthSepConv(128, 256, 2),
#           DepthSepConv(256, 256, 1),
#           DepthSepConv(256, 512, 2),
#           DepthSepConv(512, 512, 1),
#           DepthSepConv(512, 512, 1),
#           DepthSepConv(512, 512, 1),
#           DepthSepConv(512, 512, 1),
#           DepthSepConv(512, 512, 1),
#           DepthSepConv(512, 1024, 2),
#           DepthSepConv(1024, 1024, 1))


# MobileNet_V2 parameters
# BottleNeck = namedtuple('BottleNeck', ('in_c', 't', 'c', 'n', 's'))
# Conv = namedtuple('Conv', ('in_c', 'c', 'k', 's', 'p'))
# Floor = namedtuple('Floor', ('in_c', 'c', 'k', 's'))
#
# PARAMS_V2 = (
#     Conv(1, 32, 3, 2, 1),  # 224
#     BottleNeck(32, 1, 16, 1, 1),  # 112
#     BottleNeck(16, 6, 24, 2, 2),  # 112
#     BottleNeck(24, 6, 32, 3, 2),  # 56
#     BottleNeck(32, 6, 64, 4, 2),  # 28
#     BottleNeck(64, 6, 96, 3, 1),  # 14
#     BottleNeck(96, 6, 160, 3, 2),  # 14
#     BottleNeck(160, 6, 320, 1, 1),  # 7
#     Conv(320, 1280, 1, 1, 0),  # 7
#     Floor(1280, CLASSES, 1, 1)  # 1
# )

CLASSES = 10


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
    Conv(1280, CLASSES, 1, False, False, '-', 1),  # 1
)

SMALL_PARAMS = (
    Conv(1, '-', 16, 3, False, 'HS', 2),   # 224
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
    Conv(96, '-', 576, 1, True, 'HS', 1),  # 7
    Pool(576, '-', '-', '-', False, '-', 1),  # 7
    Conv(576, '-', 1024, 1, False, 'HS', 1),  # 1
    Conv(1024, '-', CLASSES, 1, False, '-', 1),  # 1
)
