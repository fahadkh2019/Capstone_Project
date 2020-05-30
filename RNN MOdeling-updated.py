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


# In[6]:


df_percent = df.sample(frac=0.4)


# In[7]:


df_percent.timestamp.dtype


# In[8]:


df_percent['timestamp'] = pd.to_datetime(df_percent['timestamp'])


# In[9]:


df_percent['timestamp'] = df_percent['timestamp'].apply(lambda x: x.strftime('%Y%m%d%H%M'))


# In[9]:


df_percent.item.dtype


# In[10]:


df_percent['item'] = df_percent['item'].astype(str)


# In[11]:


df_percent = df_percent[['timestamp','item']]


# In[12]:


df_percent_train, df_percent_test = train_test_split(df_percent, test_size=0.2)


# In[13]:


train_clicks_numpy = df_percent_train.to_numpy(dtype = 'int64') #Creating training df as numpy int64 type
test_clicks_numpy = df_percent_test.to_numpy(dtype = 'int64')  #Creating testing df as numpy int64 type


# In[14]:


featuresTrain = torch.from_numpy(train_clicks_numpy)
featuresTest = torch.from_numpy(test_clicks_numpy)


# In[16]:


print(featuresTest)


# In[17]:


# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(featuresTrain) / batch_size)
num_epochs = int(num_epochs)


# In[18]:


# Pytorch train set
train = TensorDataset(featuresTrain)


# In[19]:


# Pytorch test set
test = TensorDataset(featuresTest)


# In[20]:


# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


# In[28]:


# Create RNN
input_dim = 100    # input dimension
hidden_dim = 2  # hidden layer dimension
rnn_layer_dim = 20     # number of hidden layers
output_dim = 10   # output dimension


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = rnn_layer_dim
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, rnn_layer_dim, batch_first=True)
       
    
    def forward(self, x,hidden):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # One time step
        out, hidden = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

model = RNNModel(input_dim, hidden_dim, rnn_layer_dim, output_dim)


# In[78]:


rnn = RNNModel(input_dim,hidden_dim,layer_dim,output_dim)
print(rnn)


# In[29]:


# training the RNN
criterion = nn.CrossEntropyLoss() # loss functions used for RNN
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
seq_dim = 28  
iter = 0
for epoch in range(num_epochs):
    for i, (timestamp, item) in enumerate(train_loader):
        model.train()
        # Load images as tensors with gradient accumulation abilities
        timestamp = timestamp.view(-1, seq_dim, input_dim).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(timestamp)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, item)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 100 == 0:
            model.eval()
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for timestamp, item in test_loader:
                # Load images to a Torch tensors with gradient accumulation abilities
                timestamp = timestamp.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(timestamp)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += item.size(0)

                # Total correct predictions
                correct += (predicted == item).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


# In[25]:


len(list(model.parameters()))


# In[ ]:




