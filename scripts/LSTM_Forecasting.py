#%% import libraries/packages
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from data_prep_NucEngOverview import df

#%% import data
X = df['Nuclear Share of Electricity Net Generation']
Xy = X.values.astype(np.float32)
num_points = len(X)
num_points
#%% scaler
scaler = MinMaxScaler()
Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))

# %% Data Restructuring
X_restruct = [] 
y_restruct = [] 

for i in range(num_points-10): # i from 0 to 604
    list1 = [] # list for every i
    for j in range(i,i+10): # j from 0-10, 1-11, 2-12 (each 10 data)
        list1.append(Xy_scaled[j]) # put the Xy_scaled[0] to [9] into list1 for every i
    X_restruct.append(list1) # create X_restruct at i
    y_restruct.append(Xy_scaled[j+1]) # create y_restruct for j=10 means that the first batch will be 10 Xy_scaled data that predict the Xy_scaled data at 10

X_restruct = np.array(X_restruct)
y_restruct = np.array(y_restruct)

#%% train/test split
last_n_months = 12
clip_point = len(X_restruct) - last_n_months # the last n month will be the data to compare predicted and test value
X_train = X_restruct[:clip_point] # to clip point
X_test = X_restruct[clip_point:] # after clip point
y_train = y_restruct[:clip_point]
y_test = y_restruct[clip_point:]

print(f"X_train shape: {X_train.shape}") # OUT: (592, 10, 1) means 592 samples will be used for training, each samples has 10 datapoints
print(f"y_train shape: {y_train.shape}") # OUT: (592, 1) means that there's 592 samples of each 1 target value
print(f"X_test shape: {X_test.shape}") # OUT: (12, 10, 1)
print(f"y_test shape: {y_test.shape}") # OUT: (12, 1)


#%% Data Preparation Class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#%% dataloader
train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=2)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=len(y_test))

#%% LSTM class
class EnergyShareModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(EnergyShareModel, self).__init__()
        self.hidden_size = 150 # hardcoded inside the class
        self.lstm = nn.LSTM(input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x)    
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output

#%% Model
model = EnergyShareModel()

#%% Loss and Optimizer
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
NUM_EPOCHS = 500

#%% Training
losses = []
for epoch in range(NUM_EPOCHS):
    for j, data in enumerate(train_loader): 
        X, y = data

        # zeroes the optimizer each batch
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fun(y_pred, y)
        
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0: # each 10 epoch
        print(f"Epoch: {epoch}, Loss: {loss.data}")
        losses.append(loss.item())
    
#%% losses plot
sns.lineplot(x=range(len(losses)), y=losses)

#%% prediction
test_set = TimeSeriesDataset(X_test, y_test)
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad(): # no_grad for test data
    y_pred = model(X_test_torch) # y_prediction from X_test to model
y_act = y_test_torch.numpy().squeeze() # y_actual from y_test_torch, converted to numpy, squeezed so the shape is 1D (12, ) or array([]) not array([[]])
x_act = range(y_act.shape[0]) # get x_act from the shape of y_act 

sns.lineplot(x=x_act, y=y_act, label ='Actual', color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label ='Predicted', color='red')

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)

# %%
