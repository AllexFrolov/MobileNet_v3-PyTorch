import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from importlib import reload
import MobileNet_v3
import functions
import datafunc
import config
config = reload(config)
MobileNet_v3 = reload(MobileNet_v3)
functions = reload(functions)
datafunc = reload(datafunc)
from functions import train, accuracy
from datafunc import MyDataLoader, train_test_split
from MobileNet_v3 import get_model as gm3

device = torch.device(config.DEVICE)


if __name__ == '__main__':
    train_transformer = transforms.Compose([
        transforms.Resize(config.IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*config.NORMALIZE)
    ])

    cifar_train = datasets.CIFAR100('data/',
                                    transform=train_transformer,
                                    download=True)

    with open('classes_name.pkl', 'wb') as f:
        pickle.dump(cifar_train.classes, f)

    train_indices, val_indices = \
        train_test_split(np.arange(len(cifar_train)), .75, cifar_train.targets)
    train_loader = MyDataLoader(cifar_train, config.BATCH_SIZE, train_indices, shuffle=True)
    val_loader = MyDataLoader(cifar_train, config.BATCH_SIZE, val_indices, shuffle=True)

    mobilenet = gm3(config.MODEL_SIZE).to(device)

    optimizer = torch.optim.RMSprop(mobilenet.parameters(),
                                    lr=config.LR,
                                    momentum=config.OPTIM_MOMENTUM,
                                    weight_decay=config.WEIGHT_DECAY
                                    )
    loss_func = nn.NLLLoss()

    train_history, best_parameters = \
        train(mobilenet, train_loader, loss_func, optimizer, config.EPOCHS, accuracy, val_loader)

    torch.save({'model_state_dict': best_parameters}, 'model.torch')
