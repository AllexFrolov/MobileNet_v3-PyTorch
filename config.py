
# model parameters
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

IM_SIZE = 224
CLASSES = 10
DROPOUT = 0.2
ALPHA = 1.

