#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as torch


# In[2]:


import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# In[3]:


df = pd.read_csv("yoochoose-clicks.dat",
                     names=["session", "timestamp", "item", "category"],
                     parse_dates=["timestamp"])


# In[9]:


df_percent = df.head(50000)


# In[10]:


df_percent = df_percent[['session','item']]


# In[30]:


df_percent = df_percent.sort_values(by = 'session')


# In[35]:


test_data_size = 10004 #20 percent
train_data = df_percent[:-test_data_size]
test_data = df_percent[-test_data_size:]


# In[237]:


#getting target dataset from training dataset
target_dataset=train_data.loc[(train_data["session"]!=train_data["session"].shift(-1))]


# In[254]:


train_data['session'].isin(target_dataset['session']).value_counts()


# In[217]:


target_numpy = target_dataset.to_numpy(dtype = 'int64')


# In[109]:


train_clicks_numpy = train_data.to_numpy(dtype = 'int64') #Creating training df as numpy int64 type
test_clicks_numpy = test_data.to_numpy(dtype = 'int64')  #Creating testing df as numpy int64 type


# In[ ]:





# In[218]:


featuresTrain = torch.from_numpy(train_clicks_numpy)
featuresTest = torch.from_numpy(test_clicks_numpy)
featuresTarget = torch.from_numpy(target_numpy)


# In[114]:


# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(featuresTrain) / batch_size)
num_epochs = int(num_epochs)


# In[111]:


# Pytorch train set
train = TensorDataset(featuresTrain)


# In[112]:


# Pytorch test set
test = TensorDataset(featuresTest)


# In[115]:


# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


# In[221]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_arr = scaler.fit_transform(featuresTrain)
val_arr = scaler.transform(featuresTarget)
test_arr = scaler.transform(featuresTest)


# In[207]:



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[209]:


#####################
input_dim = 2
hidden_dim = 100
num_layers = 2 
output_dim = 1
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers,0, self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, 0, self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# In[212]:


# Train model
#####################
import numpy as np
look_back = 20
hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim =look_back-1  

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(train_inout_seq)

    loss = loss_fn(y_train_pred, train)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()


# In[ ]:




