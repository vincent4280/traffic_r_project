# __all__ = ['']

"this script store all the model that we need for DCN-based GRAND"

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv 
from torch_scatter import scatter_add


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
                          
x = torch.tensor([[5], [0], [1]], dtype=torch.float)
edge_attr = torch.tensor([2,2,3,3], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print(data.num_features)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
        #                      normalize=not args.use_gdc)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
        #                      normalize=not args.use_gdc)
        self.conv1 = ChebConv(data.num_features, data.num_features, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def diffusion_G_cal(self, x, data):
        edge_node_i_feature = x[data.edge_index[0], :]
        edge_node_j_feature = x[data.edge_index[1], :]
        edge_feature = torch.cat((edge_node_i_feature, edge_node_j_feature),dim=1)
        
        # scatter_add(x, ind, dim=0, dim_size=data)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        
        return x # F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4)], lr=0.01)  # Only perform weight-decay on first convolution.

c = model()
print(c)