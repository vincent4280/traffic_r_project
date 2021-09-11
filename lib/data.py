__all__ = ['metr_la_data']

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
    def __init__(self, data, target, transform=None):    # transform 用于更改数据
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
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
    raw_data = pd.read_hdf(r'./data/metr-la.h5').to_numpy()
    x = []; y = []
    for i in range(np.size(raw_data,0)-1):
        x.append(raw_data[i,:])
        y.append(raw_data[i+1,:])
    x = np.array(x); y = np.array(y)
    train_data_loader, test_data_loader = trainingset_construct(x_data=x, y_data=y, batch_val=100)

    return train_data_loader, test_data_loader
