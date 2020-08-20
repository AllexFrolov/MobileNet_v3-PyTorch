import torch
from tqdm import tqdm
from copy import deepcopy
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataLoader:
    def __init__(self, data, batch_size, indices=None, shuffle=False):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data = data
        if indices is None:
            self.data_len = len(data)
            self.indices = np.arange(self.data_len)
        else:
            self.data_len = len(indices)
            self.indices = indices
        self.len_ = int(np.ceil(self.data_len / batch_size))

    def __len__(self):
        return self.len_

    def create_batch(self, indices):
        batch = []
        for index in indices:
            batch.append(self.data[index][0])
        if len(batch) > 1:
            batch = torch.stack(batch)
        else:
            batch = torch.unsqueeze(batch[0], 0)
        return batch

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for n_batch in range(self.len_):
            start_index = n_batch * self.batch_size
            end_index = min(self.data_len, start_index + self.batch_size)
            batch_indices = self.indices[start_index: end_index]
            X_batch, y_batch = self.create_batch(batch_indices), \
                               self.data.targets[batch_indices]
            if len(X_batch.size()) == 3:
                X_batch = X_batch.unsqueeze(dim=1)
            yield X_batch, y_batch


def do_epoch(model, optimizer, loss_func, data_loader,
             mode='T', metric=None, title=None):
    """
    Compute one epoch
    :param model: (nn.Module) model
    :param optimizer: (torch.optim) optimization method. Ignored if mode='V'
    :param loss_func: (func) loss functions
    :param data_loader: (MyDataLoader) val batches generator (X, y). Default None
    :param mode: (str) 'T' - Train or 'V' - Validate. Default 'T'
    :param metric: (func) target metric
    :param title: (str) description in progress bar
    :return:
        epoch_loss: mean loss
        epoch_metric: mean metric
    """
    if mode not in ['V', 'T']:
        raise ValueError('mode should be "T" or "V"')
    # History
    epoch_loss = 0.
    epoch_metric = 0.

    with tqdm(total=len(data_loader)) as progress_bar:
        for ind, (X, y) in enumerate(data_loader, 1):
            description = ''
            if title is not None:
                description += f'{title: 8} |'
            description += f'Mode: {mode} |'

            X_tens, y_tens = torch.as_tensor(X, dtype=torch.float, device=DEVICE), \
                             torch.as_tensor(y, dtype=torch.long, device=DEVICE)
            predict = model(X_tens).squeeze(dim=-1)
            loss = loss_func(predict, y_tens)
            epoch_loss += loss.item()
            description += f'Loss: {epoch_loss / ind: 7.4} |'
            # backward
            if mode == 'T':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #  metric calculate
            if metric is not None:
                epoch_metric += metric(predict, y_tens)
                description += f'Metric: {epoch_metric / ind: 7.4} |'

            progress_bar.set_description(description)
            progress_bar.update()
    return epoch_loss / len(data_loader), epoch_metric / len(data_loader)


def train(model, train_loader, loss_func, optimizer, epoch_count=10,
          metric=None, val_loader=None, scheduler=None):
    """
    Training model
    :param model: (torch.nn.Module) model for train
    :param train_loader: (MyDataLoader) train batches generator (X, y)
    :param loss_func: (func) loss functions
    :param optimizer: (torch.optim) optimization method
    :param epoch_count: (int) epochs count. Default 10
    :param metric: (func) target metric
    :param val_loader: (MyDataLoader) val batches generator (X, y). Default None
    :param scheduler: (torch.utils)
    :return:
        history_info: dict of training history consist "Tloss", "Tmetric",
                    "Vloss", "Vmetric"
        best_model_param: model parameters at the highest Vmetric value
    """
    # Train_history
    history_info = {'Tloss': [], 'Tmetric': [],
                    'Vloss': [], 'Vmetric': []}
    # best Val_score and model params
    best_score = 0.
    best_model_param = {}

    datasets = {}

    if train_loader is not None:
        datasets.update({'T': train_loader})
    if val_loader is not None:
        datasets.update({'V': val_loader})

    for epoch in range(epoch_count):
        for mode, data in datasets.items():

            model.train(mode == 'T')
            epoch_loss, epoch_metric = \
                do_epoch(model, optimizer, loss_func, data,
                         mode, metric)
            history_info[mode + 'loss'].append(epoch_loss)
            history_info[mode + 'metric'].append(epoch_metric)

            if metric is not None:
                # save best metric value and model parameters
                if best_score < epoch_metric and mode == 'V':
                    best_score = epoch_metric
                    best_model_param = deepcopy(model.state_dict())

            # scheduler step
            if scheduler is not None and mode == 'V':
                scheduler.step(epoch_metric)

    return history_info, best_model_param


def accuracy(predict_proba, ground_truth):
    label_index = torch.argmax(predict_proba, dim=-1)
    true_predict = (label_index == ground_truth).sum().item()
    return true_predict / ground_truth.size()[0]
