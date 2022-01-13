from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import random


class DNNDataset(Dataset):
    def __init__(self, csv_file, window_size):
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, t):
        return self.df.iloc[t:t + self.window_size]["roberta_large_score"].values, np.expand_dims(
            self.df.iloc[t + self.window_size - 1]["direction"] > 0, axis=0)


class RNNDataset(Dataset):
    def __init__(self, csv_file, window_size):
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, t):
        return np.expand_dims(self.df.iloc[t:t + self.window_size]["roberta_large_score"].values, 1), np.expand_dims(
            np.int(self.df.iloc[t + self.window_size - 1]["direction"] > 0), axis=0)


if __name__ == '__main__':
    ds = DNNDataset("../../data/BloombergNRG_train.csv", window_size=5)
