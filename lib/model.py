# __all__ = ['']

"this script store all the model that we need for DCN-based GRAND"

import torch
from torch_geometric import data
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv 
from torch_scatter import scatter_add


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
                          
x = torch.tensor([[[2,3],[1,4],[2,3]], [[2,3],[1,4],[2,3]]], dtype=torch.float)

y = torch.tensor([[[1],[0],[1]], [[1],[0],[1]]], dtype=torch.float)
edge_attr = torch.tensor([2,2,3,3], dtype=torch.float)

# data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

class MyDataset(Dataset):
    def __init__(self, x, y, edge_index, edge_attr):    
        self.data = x
        self.target = y
        self.node_feature_dim = x.size(2)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset(x,y,edge_index, edge_attr)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

# conv1 = ChebConv(dataset.node_feature_dim, 64, K=2)
# conv2 = ChebConv(64, 1, K=2)




class Net(torch.nn.Module):
    def __init__(self, num_features, edge_index, edge_attr):
        super(Net, self).__init__()

        self.conv1 = ChebConv(num_features, 64, K=2)
        self.conv2 = ChebConv(64, 1, K=2)
        self.time_conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,3),stride=(1,1))
        self.time_conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2),stride=(1,1))
        self.time_conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2),stride=(1,1))
        self.time_conv4 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,2),stride=(1,1))
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    # def diffusion_G_cal(self, x, data):
    #     edge_node_i_feature = x[data.edge_index[0], :]
    #     edge_node_j_feature = x[data.edge_index[1], :]
    #     edge_feature = torch.cat((edge_node_i_feature, edge_node_j_feature),dim=1)
        
    #     # scatter_add(x, ind, dim=0, dim_size=data)

    def GNN_aggregation(self, x):

        x = self.conv1.forward(x, self.edge_index, self.edge_attr)
        x = self.conv2.forward(x, self.edge_index, self.edge_attr)

        return x
    
    def time_convolution(self, x):
        'description'
        # in this function, we add padding by ourselves
        x = x.unsqueeze(1)

        padding = torch.zeros((x.size(0),x.size(1),x.size(2), 1), dtype=torch.float)
        x = torch.cat((padding, x, padding), dim=3)
        x = self.time_conv1(x)
        x = self.time_conv2(x)
        x = self.time_conv3(x)
        x = self.time_conv4(x)

        return x.squeeze(1)

    def forward(self, x_list):
        'description'
        # this function is the forward of the whole architect
        'input'
        # x_list: the input of the network which is the grapg signal, graph of n timestep
        # y_list: the prediction signal of the network, graph of m timestep
        'output'
        


        'code start here'
        # using GNN for spatial correlation aggregation
        H_list = []
        for i in range(len(x_list)):
            H_list.append(self.GNN_aggregation(x_list[i]))

        # cat the feature of different timesteps, add padding by myself and add the dimension of channels
        H_time_cat = torch.cat(tuple(H_list), dim=2)
        
        # apply time convolution
        H_st = self.time_convolution(H_time_cat)

        return H_st



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= Net(2, edge_index, edge_attr).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)  # Only perform weight-decay on first convolution.

for (x,y) in loader:
    out = model.forward([x,x,x,x])
    print(out)