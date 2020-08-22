
# model parameters
MODEL_SIZE = 'small'  # should be "small" or "large"
BN_MOMENTUM = 0.99
CLASSES = 100
DROPOUT = 0.8
MIN_DEPTH = 3
ALPHA = .75  # reduce model size.

# data parameters
IM_SIZE = 128  # resize image
NORMALIZE = ([0.24, 0.25, 0.26],
             [.31, .32, .33])
BATCH_SIZE = 512

# optimizer parameters
LR = 1e-1
OPTIM_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5  # l2 weight decay

# train parameters
EPOCHS = 2
DEVICE = 'cuda'  # should be "cuda" or "cpu"

# test parameters
DATA_FOLDER = 'data/test_data/'  # for testing model
