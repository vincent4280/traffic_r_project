__all__ = ['traffic_transformer', 'transformer_training_with_schedule_sampling']


from torch._C import dtype
import torch.nn as nn
import torch
import math
import numpy as np
from torch_geometric.nn import GCNConv, ChebConv
from torch.utils.data import Dataset, DataLoader

import os
import sys
os.chdir(sys.path[0])

class traffic_transformer(nn.Module):
    def __init__(self, num_node):
        super(traffic_transformer, self).__init__()
        
        self.conv1 = ChebConv(1, 32, K=2)
        self.conv2 = ChebConv(32, 1, K=2)
        self.encoder_FC = nn.Linear(num_node, 256)    # (num_node -> feature)
        self.transformer_model = nn.Transformer(d_model=256, nhead=16, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
        self.decoder_FC = nn.Linear(256, num_node)    # (feature -> num_node)
    
    def time_pos_embedding(self, seq_length, hidden_dim):
        # t : (batch, time_sequence_len, 1)

        pos_embed = np.zeros((seq_length, hidden_dim))
        for i in range(seq_length):
            for j in range(hidden_dim):
                if j%2 == 0:
                    pos_embed[i,j] = math.sin(i/(1000)**(j/hidden_dim))
                else:
                    pos_embed[i,j] = math.cos(i/(1000)**((j-1)/hidden_dim))
        pos_embed = torch.from_numpy(pos_embed)

        return pos_embed

    def spatial_agg(self, X, edge_index, edge_weight):
        # X_encoder : (batch, Seq, node_num, feature_dim)
        # X_decoder : (batch, Seq, node_num, feature_dim)

        X = self.conv1(X, edge_index, edge_weight)
        X = self.conv2(X, edge_index, edge_weight).squeeze(-1)    # (batch, Seq, node_num)
        X = self.encoder_FC(X)    # (batch, Seq, 256)

        return X

    def prediction(self, X_encoder, edge_index, edge_weight):
        # X_encoder : (batch, Seq, node_num, feature_dim)

        seq_len = X_encoder.size(1)

        # set the decoder input
        X_false_decoder = torch.zeros_like(X_encoder)
        X_false_decoder[:,0,:,:] = X_encoder[:,-1,:,:]

        with torch.no_grad():
            for i in range(seq_len-1):

                X_next_timestep = self.forward(X_encoder, X_false_decoder, edge_index, edge_weight)[:,i,:,:]
                X_false_decoder[:,i+1,:,:] = X_next_timestep
        
        return X_false_decoder


    def forward(self, X_encoder, X_decoder, edge_index, edge_weight):

        X_encoder = self.spatial_agg(X_encoder, edge_index, edge_weight)
        X_decoder = self.spatial_agg(X_decoder, edge_index, edge_weight)
        result = self.transformer_model.forward(X_encoder, X_decoder)
        result = self.decoder_FC(result).unsqueeze(-1)

        return result

def transformer_training_with_schedule_sampling(train_loader, criterion, optimize_operator, epoch,\
    model, edge_index, edge_weight, sampling_probability):

    model.train()
    train_loss = 0
    for (encoder_input, decoder_input, GT_prediction) in train_loader:

        encoder_input = encoder_input.unsqueeze(-1)
        decoder_input = decoder_input.unsqueeze(-1)
        GT_prediction = GT_prediction.unsqueeze(-1)

        print(encoder_input.shape)

        # perform negative sampling
        with torch.no_grad():
            prediction_before_training = model.prediction(encoder_input, edge_index, edge_weight)
            new_decoder_input = torch.zeros_like(decoder_input)
            for k in range(new_decoder_input.size(1)):
                p = np.random.rand()
                if p > sampling_probability:
                    new_decoder_input[:,k,:,:] = decoder_input[:,k,:,:]
                else:
                    new_decoder_input[:,k,:,:] = prediction_before_training[:,k,:,:]

        prediction = model.forward(encoder_input, new_decoder_input, edge_index, edge_weight)
        loss = criterion(prediction, GT_prediction)
        loss.backward()
        train_loss += loss.item()
        optimize_operator.step()
    
    print('epoch:', epoch, 'loss:', train_loss)

    return model


import sys
sys.path.append(r'../')
from data import *

encoder_inputs, decoder_inputs, GT_inputs, edge_index, edge_attr = PEMS_04_for_traffic_transformer(r'../data/PEMS04', 0) 

dataset = MyDataset(encoder_inputs, decoder_inputs, GT_inputs)
loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = torch.from_numpy(edge_index.astype(int))
edge_attr = torch.from_numpy(edge_attr).float()
transformer_model = traffic_transformer(num_node=307).float()
optimizer = torch.optim.Adam(transformer_model.parameters(), weight_decay=5e-4, lr=0.01) 
criterion = nn.MSELoss()


for ep in range(1):
    transformer_model = transformer_training_with_schedule_sampling(loader, criterion, optimizer, ep,\
                        transformer_model, edge_index, edge_attr, 0.5)

    torch.save(transformer_model, r'./model.pkl')






