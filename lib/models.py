# __all__ = ['']

"this script store all the model that we need for DCN-based GRAND"

import torch
from torch._C import dtype
from torch_geometric import data
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
# from torchdyn.core.neuralde import NeuralODE
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.utils import softmax
import numpy as np


import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv 
from torch_scatter import scatter_add

import sys
sys.path.append(r'../')
from lib.data import *


x_data, y_data, t, edge_index, edge_attr = PEMS_04(r'../data/PEMS04', 0, 4)   # feature_dim=0 and timestep=4
print(x_data.shape, y_data.shape, t.shape, edge_index.shape, edge_attr.shape)
print(edge_attr)

dataset = MyDataset(x_data, y_data, t)
loader = DataLoader(dataset=dataset, batch_size=59, shuffle=False)

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''




    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials












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
    def __init__(self, num_nodes, num_features, edge_index, edge_attr, num_timestep, device):
        super(Net, self).__init__()
        'description of initialization'
        # num_nodes: number of nodes in the sensor network
        # num_feature: number of feature per sensor read
        # edge_index: (2, num_edge)
        # edge_sttr: (num_edge)
        # num_timestep: number of timestamp for sequential prediction

        self.conv1 = ChebConv(num_features, 64, K=2)
        self.conv2 = ChebConv(64, 1, K=2)
        self.time_conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,3),stride=(1,1))
        self.time_conv2 = torch.nn.Conv2d(in_channels=64, out_channels=11, kernel_size=(1,2),stride=(1,1))
      
        self.attention_layer = SpGraphTransAttentionLayer(1,2)
        self.Wout1 = nn.Linear(1, 64)
        self.Wout2 = nn.Linear(64, num_features)
        self.time_embed1 = nn.Linear(1, 128)
        self.time_embed2 = nn.Linear(128, num_nodes)

        self.num_nodes = num_nodes
        self.edge_index = edge_index.to(device)
        self.edge_attr = edge_attr.to(device)
        self.num_timestep = num_timestep
        self.device = device

    def init_state_encode(self, x, hidden_dim):
        """
        description:
          this function only serve as a function of dimension reduction, we want to obtain a hidden state to represent the
          previous traffic state, integrating spatial and temporal features

        Args:
          self.edge_index: the node index of each link, size = (2, #link)
          self.edge_attr: the distance of each link, size = (#link)
          hidden_dim: the dimension of the hidden state for each node (int)

        input:
          x: traffic state of all nodes from timestep 1 to n, size = (batchsize, #node, #time_step)

        Returns:
          x: the hidden state after the encoding, size = (#node, hidden_dim)
        """

        x = self.GNN_aggregation(x)
        






    def GNN_aggregation(self, x):

        """
        description:
          this function is used to perform convolution on the graph structure data based on the deometry graph structure

        Args:
          self.edge_index: the node index of each link, size = (2, #link)
          self.edge_attr: the distance of each link, size = (#link)
        
        input:
          x: traffic state of all nodes from timestep 1 to n, size = (batchsize, #node, #time_step)

        Returns:
          x: the hidden state after the encoding, size = (#node, #time_step)
        """

        x = self.conv1.forward(x, self.edge_index, self.edge_attr)
        x = self.conv2.forward(x, self.edge_index, self.edge_attr)

        return x
    
    def time_convolution(self, x):
        'description'
        # in this function, we add padding by ourselves
        x = x.unsqueeze(1)

        padding = torch.zeros((x.size(0),x.size(1),x.size(2), 1), dtype=torch.float).to(self.device)
        x = torch.cat((padding, x, padding), dim=3)
        x = self.time_conv1(x)
        x = self.time_conv2(x)
        x = self.time_conv3(x)
        x = self.time_conv4(x)

        return x.squeeze(1)
    
    def time_embedding(self, t):
        
        t = t.unsqueeze(0)
        t = self.time_embed1(t)
        t = self.time_embed2(t)

        return t

    def forward(self, t, x_list):
        'description'
        # this function is the forward of the whole architect
        'input'
        # x_list: [batch , (node_number*timestep) , feature_number]
        #         the input of the network which is the grapg signal, graph of n timestep
        #         these signals is cat in the second dimension
        'output'
        # partial_x_new: derivative respect to time t and state x_list

        'code start here'
        # seperate the list
        x_list = torch.chunk(input=x_list, chunks=self.num_timestep, dim=1)
        # using GNN for spatial correlation aggregation
        H_list = []
        for i in range(len(x_list)):
            H_list.append(self.GNN_aggregation(x_list[i]))

        # cat the feature of different timesteps, add padding by myself and add the dimension of channels
        H_time_cat = torch.cat(tuple(H_list), dim=2)
        
        # apply time convolution (now the timestep = 4), return (batch, num_node, 1)
        H_st = self.time_convolution(H_time_cat)

        # add time embedding effect to sensor reading
        time_embedding = self.time_embedding(t).unsqueeze(-1)
        H_st = H_st + time_embedding.repeat(H_st.size(0),1,1)

        # obtain multi-attention attribute A: [batch, n_edges, n_heads]
        A, _ = self.attention_layer(H_st, self.edge_index)

        # obtain divergent operator
        div_op = self.div_operator()

        # average the multi-head results
        partial_x_partial_t = self.diffusion_G_cal(H_st, A, div_op)

        # combine with previous time step
        dt = 1/287    # default setting if time interval is 1
        x_new = partial_x_partial_t
        partial_x_new = torch.cat(((x_list[-3]-x_list[-4])/dt, (x_list[-2]-x_list[-3])/dt, (x_list[-1]-x_list[-2])/dt, x_new), dim=1)

        return partial_x_new


x = torch.zeros((10,307,1,4)).float()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = torch.from_numpy(edge_index.astype(int))
edge_attr = torch.from_numpy(edge_attr).float()
model = Net(num_nodes=307, num_features=1, edge_index=edge_index, edge_attr=edge_attr, num_timestep=4, device=device).to(device).float()
result = model.GNN_aggregation(x)

print(result.shape)





# # define the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# edge_index = torch.from_numpy(edge_index.astype(int))
# edge_attr = torch.from_numpy(edge_attr).float()
# model = Net(num_nodes=307, num_features=1, edge_index=edge_index, edge_attr=edge_attr, num_timestep=4, device=device).to(device).float()
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.01)  # Only perform weight-decay on first convolution.
# criterion = nn.MSELoss()

# print('finish intialization')


# def train(epoch, train_loader, model, optimize_operator, criterion, device):
#     device = device
#     train_loss = 0
#     for batch_idx, (x, y, t) in enumerate(train_loader):

#         x = x.to(device)
#         y = y.to(device)
#         t = t.to(device)

#         optimize_operator.zero_grad()
#         t_span = (t[0]).squeeze(0)

#         print('calculating ODE')
#         out = odeint(model, x, t_span)
#         y_predict = out[-1]
#         loss = criterion(y_predict, y)
#         loss.backward()
#         train_loss += loss.item()
#         optimize_operator.step()

#         print('batch:', batch)

#     return model, train_loss

# for epoch in range(1):

        # with torch.no_grad():
        #     print(1)

        # model, L = train(epoch, loader, model, optimizer, criterion, device)
        # print('epoch:', epoch, 'loss:', L)

        







