# __all__ = ['']

"this script store all the model that we need for DCN-based GRAND"

import torch
from torch._C import dtype
from torch_geometric import data
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
# from torchdyn.core.neuralde import NeuralODE
import torch.nn as nn
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.utils import softmax
import numpy as np


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv 
from torch_scatter import scatter_add


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
                          
x = torch.rand((100, 12, 2))

y = torch.rand((100, 12, 2))
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
loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

# conv1 = ChebConv(dataset.node_feature_dim, 64, K=2)
# conv2 = ChebConv(64, 1, K=2)


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = 0.2   # user-defined
    self.concat = concat
    self.h = 8    # user_defined
    self.edge_weights = edge_weights
    self.softmax_fun = nn.Softmax(dim=1)

    try:
      self.attention_dim = 128    # user-defined
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    self.Q = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.Q)

    self.V = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.V)

    self.K = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):

    num_node = x.size(1)

    q = self.Q(x)    # (batch * node_number * attention_number)
    k = self.K(x)    # (batch * node_number * attention_number)
    v = self.V(x)    # (batch * node_number * attention_number)

    # perform linear operation and split into h heads

    k = k.view(-1, num_node, self.h, self.d_k)
    q = q.view(-1, num_node, self.h, self.d_k)
    v = v.view(-1, num_node, self.h, self.d_k)

    # transpose to get dimensions [batch, n_nodes, attention_dim, n_heads]

    k = k.transpose(2, 3)
    q = q.transpose(2, 3)
    v = v.transpose(2, 3)

    src = q[:, edge[0, :], :, :]
    dst_k = k[:, edge[1, :], :, :]
    prods = torch.sum(src * dst_k, dim=2) / np.sqrt(self.d_k)    # [batch, n_edges, n_heads]

    # if self.opt['reweight_attention'] and self.edge_weights is not None:
    #   prods = prods * self.edge_weights.unsqueeze(dim=1)
    attention = self.softmax_fun(prods)    #  softmax in dimension 1 to obtain [batch, n_edges, n_heads]

    return attention, v

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class Net(torch.nn.Module):
    def __init__(self, num_nodes, num_features, edge_index, edge_attr, num_timestep):
        super(Net, self).__init__()

        self.conv1 = ChebConv(num_features, 64, K=2)
        self.conv2 = ChebConv(64, 1, K=2)
        self.time_conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,3),stride=(1,1))
        self.time_conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2),stride=(1,1))
        self.time_conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,2),stride=(1,1))
        self.time_conv4 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,2),stride=(1,1))
        self.attention_layer = SpGraphTransAttentionLayer(1,2)
        self.Wout1 = nn.Linear(1, 64)
        self.Wout2 = nn.Linear(64, num_features)

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_timestep = num_timestep

    def div_operator(self):

        div_op = np.zeros((self.num_nodes, self.edge_index.size(1)))
        for i_index in range(self.num_nodes):
            for k in range(self.edge_index.size(1)):
                if self.edge_index[0,k] == i_index:
                    j_index = self.edge_index[1,k]
                    div_op[i_index, j_index] = self.edge_attr[j_index]
        div_op = torch.from_numpy(div_op).float()

        return div_op


    def diffusion_G_cal(self, x, A, div_op):

        num_node = div_op.size(0)
        num_edge = div_op.size(1)
        num_head = A.size(2)

        
        edge_node_i_feature = x[:, self.edge_index[0], :]
        edge_node_j_feature = x[:, self.edge_index[1], :]
        edge_feature_diff = edge_node_i_feature - edge_node_j_feature    # (batch, num_edge, 1)

        Gx = A * torch.repeat_interleave(input=edge_feature_diff, repeats=num_head, dim=2)    # (batch, num_edge, n_head)
        GGx = torch.mm(div_op, Gx.view(num_edge,-1))    
        GGx = GGx.view(-1, num_node, num_head)    # (batch, num_node, n_head)
        GGx = torch.mean(GGx, dim=2).unsqueeze(-1)    # (batch, num_node, 1)
        GGx = self.Wout2(self.Wout1(GGx))

        return GGx

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

    def forward(self, t, x_list):
        'description'
        # this function is the forward of the whole architect
        'input'
        # x_list: [batch * (node_number*timestep) * feature_number]
        #         the input of the network which is the grapg signal, graph of n timestep
        #         these signals is cat in the second dimension
        'output'
        

        'code start here'
        # seperate the list
        x_list = torch.chunk(input=x_list, chunks=self.num_timestep, dim=1)
        # using GNN for spatial correlation aggregation
        H_list = []
        for i in range(len(x_list)):
            H_list.append(self.GNN_aggregation(x_list[i]))

        # cat the feature of different timesteps, add padding by myself and add the dimension of channels
        H_time_cat = torch.cat(tuple(H_list), dim=2)
        
        # apply time convolution (now the timestep = 4), return (batch * num_node * 1)
        H_st = self.time_convolution(H_time_cat)

        # obtain multi-attention attribute A: [batch, n_edges, n_heads]
        A, _ = self.attention_layer(H_st, self.edge_index)

        # obtain divergent operator
        div_op = self.div_operator()

        # average the multi-head results
        partial_x_partial_t = self.diffusion_G_cal(H_st, A, div_op)

        # combine with previous time step
        x_new = x_list[-1] + partial_x_partial_t
        x_new = torch.cat((x_list[-3], x_list[-2], x_list[-1], x_new), dim=1)

        return x_new



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, 2, edge_index, edge_attr, 4).to(device)
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)  # Only perform weight-decay on first convolution.


for (x,y) in loader:
    # out = model.forward(1,x)
    out = odeint(model, x, torch.tensor([0,0.25,0.5,0.75,1], dtype=torch.float64))
    print(out[-1])

    break


