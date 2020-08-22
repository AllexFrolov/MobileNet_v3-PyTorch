import numpy as np
import torch


def train_test_split(data, train_size, stratify=None):
    if stratify is None:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(len(indices) * train_size)
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        return data[train_indices], data[test_indices]
    else:
        unique_values = np.unique(stratify)
        train_data = []
        test_data = []
        for u_value in unique_values:
            u_train, u_test = train_test_split(data[stratify == u_value], train_size)
            train_data.append(u_train)
            test_data.append(u_test)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return np.concatenate(train_data), np.concatenate(test_data)


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
        X_batch = []
        y_batch = []
        for index in indices:
            X, y = self.data[index]
            X_batch.append(X)
            y_batch.append(y)
        if len(X_batch) > 1:
            X_batch = torch.stack(X_batch)
        else:
            X_batch = torch.unsqueeze(X_batch[0], 0)
        return X_batch, y_batch

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for n_batch in range(self.len_):
            start_index = n_batch * self.batch_size
            end_index = min(self.data_len, start_index + self.batch_size)
            batch_indices = self.indices[start_index: end_index]
            X_batch, y_batch = self.create_batch(batch_indices)
            yield X_batch, y_batch
