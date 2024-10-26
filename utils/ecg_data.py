import os
import sys
import re
import glob
import pickle
import copy
import torch
import wfdb
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from Build_Graph.build_Graph2 import *


class ECGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["records100"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        rawdata_path = "./data/cad/raw/"
        X, label, _ = load_dataset(rawdata_path, 100)
        X = X.transpose(0, 2, 1)
        edge_indexs = create_edge_index2(X)
        for idx in range(X.shape[0]):
            data = Data(x=torch.tensor(X[idx]), edge_index=edge_indexs[idx], y=label[idx])
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices),
                   self.processed_paths[0])



def select_dataset(dataset, Y):
    Y.index = range(len(Y))
    train_dataset = dataset[list(Y[Y.strat_fold == 1].index)]
    val_dataset = dataset[list(Y[Y.strat_fold == 2].index)]
    test_dataset = dataset[list(Y[Y.strat_fold == 3].index)]
    return train_dataset, val_dataset, test_dataset


def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-3] == 'cad':
        Y = pd.read_csv(path + 'CAD.csv', index_col='ecg_id')
        X = load_raw_data_cad(Y, sampling_rate, path)
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        label = lb.fit_transform(Y["scp_codes"].values)
        labels = np.concatenate((1 - label, label), axis=1)
    else:
        print("invalid dataset")

    return X, labels, Y



def load_raw_data_cad(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            csv_files = [path + str(f) + '.csv' for f in tqdm(df.index)]
            data = [pd.read_csv(file).values for file in tqdm(csv_files)]
            data_array = np.array(data)
            pickle.dump(data_array, open(path + 'raw100.npy', 'wb'), protocol=4)
    else:
        print("samplerate!=100")

    return data

