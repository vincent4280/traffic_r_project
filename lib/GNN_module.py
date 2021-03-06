__all__ = ['cheb_polynomial', 'Spatial_Attention_layer', 'cheb_conv_with_SAt', 'cheb_conv']

import numpy as np
import torch.nn as nn
import torch 
import torch.nn.functional as F


def cheb_polynomial(L_tilde, K, device):
    '''
    description: compute a list of chebyshev polynomials from T_0 to T_{K-1} 

    package: numpy

    Parameters:
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials

    Returns:
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(device).float(), torch.from_numpy(L_tilde).to(device).float()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

class Spatial_Attention_layer(nn.Module):
    '''
    description: compute spatial attention scores

    Args:
    # num_time_step: dimension of time, it should be init_hidden_state_dim + 1
    # num_of_features: dimension of features
    # num_of_nodes: number of nodes

    forward function input:
    # x: (batch, #node, #feature, init_hidden_dim + 1)

    forward function output
    # S_normalized: (batch, #node, #node)
    '''
    def __init__(self, num_time_step, num_of_features, num_of_nodes):
        super(Spatial_Attention_layer, self).__init__()
        
        self.W_1 = nn.Parameter(torch.ones(num_time_step))
        self.W_2 = nn.Parameter(torch.ones((num_of_features, num_time_step)))
        self.W_3 = nn.Parameter(torch.ones(num_of_features))
        self.b_s = nn.Parameter(torch.ones((1, num_of_nodes, num_of_nodes)))
        self.V_s = nn.Parameter(torch.ones((num_of_nodes, num_of_nodes)))
    
    def batch_dot(self, x1, x2):
        '''
        description: this function is used to batch dot two tensors
        (batch, N, K) * (batch, K, M) --> (batch, N, M)

        input:
        x1: (batch, N, K)
        x2: (batch, K, M)

        output:
        y: (batch, N, M)

        '''

        batch_num = x1.shape[0]
        y_list = []
        for i in range(batch_num):
            y_list.append(torch.mm(x1[i], x2[i]).unsqueeze(0))
        y = torch.cat(y_list,dim=0)

        return y

    def forward(self, x):

        # compute spatial attention scores
        # shape of lhs is (batch_size, #node, #time_step)

        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)

        # after transpose (#feature_per_node, batch_size, #time_step, #node)
        # shape of rhs is (batch_size, #time_step, #node)
        rhs = torch.matmul(x.permute(0, 1, 3, 2), self.W_3).permute(0,2,1)

        # shape of product is (batch_size, #node, #node)
        product = self.batch_dot(lhs, rhs)

        # shape of S is (batch_size, #node, #node)
        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s).permute(1, 2, 0)).permute(2, 0, 1)

        # normalization: shape of S is (batch_size, #node, #node)
        S = S - torch.max(S, dim=1, keepdims=True)[0]
        exp = torch.exp(S)

        S_normalized = exp / (torch.sum(exp, dim=1, keepdims=True)[0])
        return S_normalized

class cheb_conv_with_SAt(nn.Module):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''
    def __init__(self, features_in_dim, features_out_dim, K, cheb_polynomials):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials will be used in this convolution. It defines the neighbourhood distance of node aggragration.
        '''

        super(cheb_conv_with_SAt, self).__init__()
        self.K = K
        self.num_of_filters = 64
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.ones((self.K, features_in_dim, self.num_of_filters)))
        self.Beta = nn.Parameter(torch.ones((self.K, self.num_of_filters, features_out_dim)))
        self.activation = nn.ReLU()

    def batch_dot(self, x1, x2):
        '''
        description: this function is used to batch dot two tensors
        (batch, N, K) * (batch, K, M) --> (batch, N, M)

        input:
        x1: (batch, N, K)
        x2: (batch, K, M)

        output:
        y: (batch, N, M)

        '''

        batch_num = x1.shape[0]
        y_list = []
        for i in range(batch_num):
            y_list.append(torch.matmul((x1[i]), (x2[i])).unsqueeze(0))
        y = torch.cat(y_list,dim=0)

        return y

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, #node, #feature_per_node, #time_step)

        spatial_attention: mx.ndarray, shape is (batch_size, #node, #node)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, #node, self.num_of_filters, #time_step)
        '''

        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, #node, #feature_per_node)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices, num_of_features))
            
            for k in range(self.K):

                # shape of T_k is (#node, #node)
                T_k = (self.cheb_polynomials[k]).unsqueeze(0).repeat(batch_size,1,1)

                # shape of T_k_with_attention is (batch_size, #node, #node)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (#feature_per_node, num_of_filters)
                theta_k = self.Theta[k]
                beta_k = self.Beta[k]

                # shape is (batch_size, #node, #feature_per_node)
                rhs = self.batch_dot(T_k_with_at.permute(0, 2, 1), graph_signal)

                # shape is (batch_size, #node, num_of_filters)
                Ir = self.activation(torch.matmul(rhs, theta_k))
                output = output + torch.matmul(Ir, beta_k)

            outputs.append(output.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)

class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution without Spatial Attention scores
    '''
    def __init__(self, features_in_dim, features_out_dim, K, cheb_polynomials):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials will be used in this convolution. It defines the neighbourhood distance of node aggragration.
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.num_of_filters = 64    # similar to hidden layer
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.ones((self.K, features_in_dim, self.num_of_filters)))
        self.Beta = nn.Parameter(torch.ones((self.K, self.num_of_filters, features_out_dim)))
        self.activation = nn.ReLU()

    def batch_dot(self, x1, x2):
        '''
        description: this function is used to batch dot two tensors
        (batch, N, K) * (batch, K, M) --> (batch, N, M)

        input:
        x1: (batch, N, K)
        x2: (batch, K, M)

        output:
        y: (batch, N, M)

        '''

        batch_num = x2.shape[0]
        y_list = []
        for i in range(batch_num):
            y_list.append(torch.matmul((x1[i]), (x2[i])).unsqueeze(0))
        y = torch.cat(y_list,dim=0)

        return y

    def forward(self, x):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, #node, #feature_per_node, #time_step)

        spatial_attention: mx.ndarray, shape is (batch_size, #node, #node)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, #node, feature_out_dim, #time_step)
        '''

        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, #node, #feature_per_node)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices, num_of_features))
            
            for k in range(self.K):

                # shape of T_k is (batch, #node, #node)
                T_k = (self.cheb_polynomials[k]).unsqueeze(0).repeat(batch_size,1,1)

                # shape of theta_k is (#feature_per_node, num_of_filters)
                theta_k = self.Theta[k]
                beta_k = self.Beta[k]

                # shape is (batch_size, #node, #feature_per_node)
                rhs = self.batch_dot(T_k.permute(0,2,1), graph_signal)

                # shape is (batch_size, #node, num_of_filters)
                Ir = self.activation(torch.matmul(rhs, theta_k))
                output = output + torch.matmul(Ir, beta_k)

            outputs.append(output.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)