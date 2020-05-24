#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch as torch


# In[4]:


import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# In[5]:


df = pd.read_csv("yoochoose-clicks.dat",
                     names=["session", "timestamp", "item", "category"],
                     parse_dates=["timestamp"])


# In[82]:


df_test = pd.read_csv("yoochoose-test.dat",
                     names=["session", "timestamp", "item", "category"],
                     parse_dates=["timestamp"])


# In[6]:


df_percent = df.sample(frac=0.2)


# In[17]:


df_percent.timestamp.dtype


# In[43]:


df_percent['timestamp'] = pd.to_datetime(df_percent['timestamp'])


# In[44]:


df_percent['timestamp'] = df_percent['timestamp'].apply(lambda x: x.strftime('%Y%m%d%H%M'))


# In[27]:


df_percent.item.dtype


# In[26]:


df_percent['item'] = df_percent['item'].astype(str)


# In[45]:


df_percent = df_percent[['timestamp','item']]


# In[49]:


clicks_numpy = df_percent.to_numpy(dtype = 'int64')


# In[50]:


featuresTrain = torch.from_numpy(clicks_numpy)


# In[51]:


print(featuresTrain)


# In[52]:


# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(featuresTrain) / batch_size)
num_epochs = int(num_epochs)


# In[53]:


# Pytorch train set
train = TensorDataset(featuresTrain)


# In[54]:


# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)


# In[69]:


# Create RNN
input_dim = 100    # input dimension
hidden_dim = 2  # hidden layer dimension
layer_dim = 20     # number of hidden layers
output_dim = 10   # output dimension


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.embeddings = nn.embeddings(input_dim , output_dim)
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
       
    
    def forward(self, x,hidden):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).requires_grad_()
        # One time step
        emb = self.embeddings(x)
        out, hidden = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)


# In[78]:


rnn = RNNModel(input_dim,hidden_dim,layer_dim,output_dim)
print(rnn)


# In[80]:


# training the RNN
criterion = nn.CrossEntropyLoss() # loss functions used for RNN
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
seq_dim = 28  


# In[ ]:




