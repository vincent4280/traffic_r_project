import sys, os
sys.path.append(r'../')

# for debug
os.chdir(sys.path[0])

from lib.ASTODE import *
from lib.data import *

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np

# load the data
x, y, t, adj = PEMS_04(r'../../data/PEMS04', 12, 1)
x = np.expand_dims((x[:,:,0,:]), axis=2)   # pick the first feature
y = np.expand_dims((y[:,:,0,:]), axis=2)    # pick the first feature
train_dataset = MyDataset(x[0:int(0.7*x.shape[0])], y[0:int(0.7*y.shape[0])], t[0:int(0.7*t.shape[0])])
test_dataset = MyDataset(x[int(0.8*x.shape[0]):int(0.99*x.shape[0])], y[int(0.8*y.shape[0]):int(0.99*y.shape[0])], t[int(0.8*t.shape[0]):int(0.99*t.shape[0])])
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

# define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for batch_idx, (x, y, t) in enumerate(train_loader):
    x_example = x
    y_example = y
    predict_time_steps = y.shape[-1]
    break

model = AST_GODE(x=x_example, adj=adj, K=2, temporal_hidden_dim=3, device=device).to(device).float()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=0.001)  
criterion = nn.MSELoss()


print('finish intialization')

def train(epoch, train_loader, model, optimize_operator, criterion, num_eval_point, device):
    train_loss = 0
    for batch_idx, (x, y, t) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        t = t.to(device)

        print('calculating ODE')
        optimize_operator.zero_grad()
        t_span = torch.linspace(0, 0.1*num_eval_point, num_eval_point+1).float()
        yhat = model.forward(x=x, T=t, t_span=t_span)

        loss = criterion(yhat, y)
        loss.backward()
        train_loss += loss.item()
        optimize_operator.step()

    return model, train_loss

def test(epoch, test_loader, model, num_eval_point, device):

    with torch.no_grad():
        test_loss = 0
        for batch_idx, (x, y, t) in enumerate(test_loader):

            x = x.to(device)
            t_span = torch.linspace(0, 0.1*num_eval_point, num_eval_point+1).float()
            yhat = model.forward(x=x, T=t, t_span=t_span)

            print('calculating ODE')
            loss = criterion(yhat, y)
            test_loss += loss.item()
    
    return test_loss


for time_step in range(0,12):

    x, y, t, adj = PEMS_04(r'../../data/PEMS04', 12, time_step+1)
    x = np.expand_dims((x[:,:,0,:]), axis=2)   # pick the first feature
    y = np.expand_dims((y[:,:,0,:]), axis=2)    # pick the first feature

    # construct the training data loader for "n time_step" prediction
    train_dataset = MyDataset(x[0:int(0.7*x.shape[0])], y[0:int(0.7*y.shape[0])], t[0:int(0.7*t.shape[0])])
    test_dataset = MyDataset(x[int(0.8*x.shape[0]):int(0.99*x.shape[0])], y[int(0.8*y.shape[0]):int(0.99*y.shape[0])], t[int(0.8*t.shape[0]):int(0.99*t.shape[0])])
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    for epoch in range(1000):

        model, train_loss = train(epoch, train_loader, model, optimizer, criterion, time_step+1, device)

        if epoch % 100 == 0:
            torch.save(model, r'./model.pkl')

        if epoch % 100 == 0:
            test_loss = test(epoch, test_loader, model, time_step+1, device)
            print('epoch:', epoch, 'training loss:', train_loss, 'test_loss:', test_loss)

        


