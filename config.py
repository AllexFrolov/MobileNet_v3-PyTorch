
# model parameters
MODEL_SIZE = 'small'
BN_MOMENTUM = 0.99
CLASSES = 100
DROPOUT = 0.8
MIN_DEPTH = 3
ALPHA = .5

# data parameters
IM_SIZE = 128
NORMALIZE = ([0.24, 0.25, 0.26],
             [.31, .32, .33])
BATCH_SIZE = 32

# optimizer parameters
LR = 1e-1
OPTIM_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

# train parameters
EPOCHS = 3
DEVICE = 'cuda'

# test parameters
DATA_FOLDER = 'data/test_data/'