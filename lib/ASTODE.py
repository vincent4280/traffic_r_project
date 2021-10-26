__all__ = ['AST_GODE']

import torch.nn as nn
import torchdiffeq
import torch
import sys, os

# for debug
os.chdir(sys.path[0])
from lib.GNN_module import *


class init_hidden_state_encoder(nn.Module):

    '''
    description: this class is to calculate the initial hidden state of the input data

    Args:
    # cheb_polynomial: list of 2D tensor, each 2D tensor is of shape of (#node, #node)
    # temporal_input_dim: temporal dimension of the graph signals
    # temporal_hidden_dim: output dimension of the initial hidden state

    forward function input:
    # x: traffic state data (batch, #node, #feature, temporal_input_dim)

    forward function output:
    # x: traffic state data (batch, #node, #feature, temporal_hidden_dim)    
    '''

    def __init__(self, cheb_polynomial, temporal_input_dim, temporal_hidden_dim, num_of_features):
        super(init_hidden_state_encoder, self).__init__()

        self.K = len(cheb_polynomial)
        self.GNN_layer = cheb_conv(features_in_dim=num_of_features, features_out_dim=num_of_features, K=self.K, cheb_polynomials=cheb_polynomial)

        self.time_conv1 = nn.Linear(in_features=temporal_input_dim, out_features=64)
        self.time_conv2 = nn.Linear(in_features=64, out_features=temporal_hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, T):
        '''
        p.s. by now we have not implement the initial time embedding
        '''

        # output is (batch, #node, #feature, #time_step)
        x_after_Spatial_agg = self.GNN_layer.forward(x)

        # output is (batch, #node, #feature, temporal_hidden_dim)
        x_after_ST_agg = self.time_conv2(self.activation(self.time_conv1(x_after_Spatial_agg)))

        return x_after_ST_agg

class ode_derivative_fun(nn.Module):

    '''
    description: this class is to calculate the derivative of the state with respect to time of any given time_stamp

    Args:
    # cheb_polynomial: list of 2D tensor, each 2D tensor is of shape of (#node, #node)
    # num_of_nodes: number of nodes of the graph

    forward function input:
    x: traffic state data (batch, #node, feature_dim, init_hidden_dim + 1)
        where init_hidden_dim is the dimension of the hidden state
    t: time_stamp that current state is (0-D tensor)

    forward function output:
    derivative: traffic state data derivative (batch, #node, #feature, 1)    
    '''

    def __init__(self, cheb_polynomials, num_of_nodes, num_of_features, num_time_step):
        super(ode_derivative_fun, self).__init__()

        K = len(cheb_polynomials)
        self.time_embed = nn.Linear(1, num_of_nodes)
        self.att_layer = cheb_conv_with_SAt(features_in_dim=num_of_features, features_out_dim=num_of_features, K=K, cheb_polynomials=cheb_polynomials)
        self.Spatial_Attention_score = Spatial_Attention_layer(num_time_step=num_time_step, num_of_features=num_of_features, num_of_nodes=num_of_nodes)

    def forward(self, t, x):
        
        (batchsize, num_nodes, feature_dim, time_stamp) = x.shape

        # encode the time_step information into x, output is (batch, #node, #feature, #time_stamp)
        t = t.reshape(1,1).repeat(batchsize,1)
        time_info = self.time_embed(t).unsqueeze(-1).unsqueeze(-1).repeat(1,1,feature_dim, time_stamp)
        x = x + time_info

        spatial_attention = self.Spatial_Attention_score(x)
        derivative = self.att_layer(x, spatial_attention)

        return derivative.squeeze(-1)

class GDEFunc(nn.Module):
    def __init__(self, gnn:nn.Module):
        """
        description: General GDE function class. To be passed to an ODEBlock
        
        forward function input:
        x: current traffic state, not the hidden state
        t: current time step

        forward function output:
        x: after the operation of gnn we can obtain the derivative of the state with respect to time
        
        """
        super().__init__()
        self.gnn = gnn
        self.nfe = 0
            
    def forward(self, t, x):   # 模拟derivative的项与t无关
        self.nfe += 1
        x = self.gnn(x)
        return x

class ControlledGDEFunc(nn.Module):
    def __init__(self, x, adj, K, temporal_input_dim, temporal_hidden_dim):

        """
        description: Controlled GDE function class. To be passed to an ODEBlock
                     Input information is preserved longer via hooks to input node features X_0, affecting all ODE function steps. 
                     Requires assignment of '.h0' before calling .forward
        
        Args:
        # x: input graph signal (batch, #node, #feature, temporal_input_dim)
        # adj: adjacent matrix (#node, #node)
        # K: the order of neighbour of the chebshev convolution
        # temporal_input_dim: temporal input dimension of the graph signal
        # temporal_hidden_dim: temporal dimension of the hidden initial state
        
        forward function input:
        # x: current traffic state, not the hidden state (batch, #node, #feature)
        # t: current time step (0-D tensor)

        forward function output:
        # x: after the operation of gnn we can obtain the derivative of the state with respect to time
        
        """

        super(ControlledGDEFunc, self).__init__()
        self.nfe = 0

        (batchsize, num_nodes, feature_dim, time_stamp) = x.shape

        cheb_polynomials = cheb_polynomial(adj, K)
        self.Hinit_encoder = init_hidden_state_encoder(cheb_polynomials, temporal_input_dim, temporal_hidden_dim, feature_dim)
        self.derivative_calculator = ode_derivative_fun(cheb_polynomials, num_nodes, feature_dim, temporal_hidden_dim + 1)
    
    def forward(self, t, x):
        self.nfe += 1
        x = x.unsqueeze(-1)
        x = torch.cat([x, self.h0], dim=-1)   
        derivative = self.derivative_calculator.forward(x=x, t=t)

        return derivative

    def init_hidden(self, x, T):
        self.h0 = self.Hinit_encoder.forward(x=x, T=T)


class AST_GODE(nn.Module):

    """ 
    description: Attention based graph neural ODE
    
    Args:
    # x: input graph signal (batch, #node, #feature, #time_step)
    # T: input time information (batch, 1)
    # adj: adjacent matrix (#node, #node)
    # K: hyper-parameter of the chebshev graph convolution
    # temporal_hidden_dim: hidden dimension of the inital hidden state
    # method:str = {'euler', 'rk4', 'dopri5', 'adams'}
    # rtol: relative error tolerance
    # atol: absolute error tolerance
    # adjoint: binary variable as a flag whether using adjoint method

    forward function input:
    # x: current traffic state, initial state of the forward (batch, #node, #feature, 1)
    # T: evaluation point of neural ODE (1D tensor)

    forward function output
    # out[1:(out.shape[0])]: 5D tensor (evaluation_time-1, batch, #node, #feature, 1)
    """

    def __init__(self, x, adj, K, temporal_hidden_dim, method:str='euler', rtol:float=1e-3, atol:float=1e-4, adjoint:bool=True):
        super(AST_GODE, self).__init__()

        temporal_input_dim = x.shape[3]
        self.odefunc = ControlledGDEFunc(x, adj, K, temporal_input_dim, temporal_hidden_dim)
        
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint_flag = adjoint

    def forward(self, x:torch.Tensor, T, t_span:torch.Tensor):

        # perform ST_encoding for input gragh signal
        self.odefunc.init_hidden(x, T)

        # obtain integration time_stamp
        self.integration_time = t_span.float()
        self.integration_time = self.integration_time.type_as(x)   # 转换t_interval 为与x相同的数据结构
        
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x[:,:,:,-1], self.integration_time, rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x[:,:,:,-1], self.integration_time, rtol=self.rtol, atol=self.atol, method=self.method)

        return out[1:(out.shape[0])]    # output the result of the evaluation points
    
    def forward_batched(self, x:torch.Tensor, nn:int, indices:list, timestamps:set):
        """ Modified forward for ODE batches with different integration times """

        timestamps = torch.Tensor(list(timestamps))
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, timestamps,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, timestamps,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        out = self._build_batch(out, nn, indices).reshape(x.shape)
        return out
        
    
    def _build_batch(self, odeout, nn, indices):
        b_out = []
        for i in range(len(indices)):
            b_out.append(odeout[indices[i],i*nn:(i+1)*nn])
        return torch.cat(b_out).to(odeout.device)
              
        
    def trajectory(self, x:torch.Tensor, T:int, num_points:int):
        self.integration_time = torch.linspace(0, T, num_points)
        self.integration_time = self.integration_time.type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                 rtol=self.rtol, atol=self.atol, method=self.method)
        return out