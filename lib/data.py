__all__ = ['metr_la_data', 'PEMS_04', 'MyDataset', 'PEMS_04_for_traffic_transformer']

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

def training_loader_construct(dataset,batch_num):
    'description'
    # construct the train loader given the dataset and batch size value
    # this function can be used for all different cases 

    train_loader = DataLoader(
        dataset,
        batch_size=batch_num,
        shuffle=True,                     # change the sequence of the data every time
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader


class MyDataset(Dataset):
    def __init__(self, x, y, t):    
        self.data = torch.from_numpy(x).float()
        self.target = torch.from_numpy(y).float()
        self.timestamp = torch.from_numpy(t).float()
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        t = self.timestamp[index]

        return x, y, t
    
    def __len__(self):
        return len(self.data)


def trainingset_construct(x_data, y_data, batch_val):
    dataset = MyDataset(x_data, y_data)
    train_number = int(0.8 * np.size(x_data,0))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_number, np.size(x_data,0) - train_number])
    train_data_loader = training_loader_construct(dataset = train_dataset, batch_num = batch_val)
    test_data_loader = training_loader_construct(dataset = test_dataset, batch_num = batch_val)
    return train_data_loader, test_data_loader


def metr_la_data(path):
    raw_data = pd.read_hdf(path).to_numpy()
    x = []; y = []
    for i in range(np.size(raw_data,0)-1):
        x.append(raw_data[i,:])
        y.append(raw_data[i+1,:])
    x = np.array(x); y = np.array(y)
    train_data_loader, test_data_loader = trainingset_construct(x_data=x, y_data=y, batch_val=100)

    return train_data_loader, test_data_loader

def PEMS_04(path, n):
    'description'
    # this function is used for constructing the training data loader of PEMS data
    # the dataset has 59 days data, every 5 minutes per reading, 307 sensors, 3 features (PEMS04)
    'input'
    # path: the local path of the dataset
    # raw_data: (time_stamp, #node, feature_dim)
    # feature_dim: choose one of the feature for prediction 0: flow, 1: speed, 2: occupancy
    # n: number of timestamp used for prediction of next time step

    # load the sensor data
    raw_data = np.load(path + '/pems04.npz', allow_pickle=False)['data']
    
    # calculate network adj matrix
    network_data = np.genfromtxt(path + '/distance.csv', delimiter=',')
    network_data = network_data[1:np.size(network_data,0),:]
    edge_index = np.transpose(network_data)[0:2, :]
    edge_attr = np.transpose(network_data)[2, :]
    num_nodes = np.size(raw_data,1)
    adj = np.zeros((num_nodes, num_nodes))
    for k in range(np.size(edge_attr)):
        adj[edge_index[0,k], edge_index[1,k]] = edge_attr[k]
        adj[edge_index[1,k], edge_index[0,k]] = edge_attr[k]

    # store the data
    x = []; y = []; t = []

    num_datapoints = np.size(raw_data,0)
    reading_per_day =  24 * 60 / 5
    Time_stamp = np.linspace(0, 10, int(reading_per_day), endpoint=True)

    for i in range(num_datapoints - 3*n):
        x.append(raw_data[i:i+n, :, :])
        y.append(raw_data[i+n-1:i+2*n-1, :, :])
        t.append(Time_stamp[int(i%reading_per_day)])
    x = np.array(x); y = np.array(y); t = np.array(t)

    return x, y, t, adj

    


def PEMS_04_for_traffic_transformer(path, feature_dim):
    'description'
    # this function is used for constructing the training data loader of PEMS data
    # the dataset has 59 days data, every 5 minutes per reading, 307 sensors, 3 features (PEMS04)
    'input'
    # path: the local path of the dataset
    # feature_dim: choose one of the feature for prediction 0: flow, 1: speed, 2: occupancy
    # n: number of timestamp used for prediction of next time step

    raw_data = np.load(path + '/pems04.npz', allow_pickle=False)['data']
    # network_data = pd.read_csv(path + '/distance.csv', header=None).to_numpy()
    network_data = np.genfromtxt(path + '/distance.csv', delimiter=',')
    network_data = network_data[1:np.size(network_data,0),:]
    edge_index = np.transpose(network_data)[0:2, :]
    edge_attr = np.transpose(network_data)[2, :]
    encoder_inputs = []; decoder_inputs = []; GT_inputs = []

    for i in range(raw_data.shape[0] - 25):

        encoder_inputs.append(raw_data[i:i+12, :, feature_dim])
        decoder_inputs.append(raw_data[i+11:i+23, :, feature_dim])
        GT_inputs.append(raw_data[i+12:i+24, :, feature_dim])
    
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    GT_inputs = np.array(GT_inputs)

    return encoder_inputs, decoder_inputs, GT_inputs, edge_index, edge_attr


