# magic dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')
import random
import numpy as np
import pandas as pd
from math import sqrt
import math
import os
import random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from autoimpute.imputations import MultipleImputer

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Autoencoder, self).__init__()
        self.dim = dim
        
        self.drop_out = nn.Dropout(p=0.2)
        
        # encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(dim, int(dim*0.7)),
            nn.Tanh(),
            nn.Linear(int(dim*0.7), int(dim*0.5)),
            nn.Tanh(),
            nn.Linear(int(dim*0.5), int(dim*0.2))
        )
        
        # decoder architecture
        self.decoder = nn.Sequential(
            nn.Linear(int(dim*0.2), int(dim*0.5)),
            nn.Tanh(),
            nn.Linear(int(dim*0.5), int(dim*0.7)),
            nn.Tanh(),
            nn.Linear(int(dim*0.7), dim)
        )
        
    def forward(self, x):
        #print("x shape:", x.shape)
        x = x.view(-1, self.dim)
        
        # adding dropout to introduce input corruption during training
        x_missed = self.drop_out(x)
        
        z = self.encoder(x_missed)
        out = self.decoder(z)
        
        out = out.view(-1, self.dim)
        
        return out
    
def training(num_epochs, model, train_loader, criterion, optimizer):
  for epoch in range(num_epochs):
      loss = 0
      for i, batch_features in enumerate(train_loader):
          # load it to the active device
        #   batch_features = batch_features.to(device)
          batch_features = batch_features.float().to(device)
        #   print(" == batch features === ", type(batch_features))

          # reset the gradients back to zero
          optimizer.zero_grad()
        
          # compute reconstructions
          outputs = model(batch_features)
        #   print("== outputs ===", outputs)
        
          # compute training reconstruction loss
          train_loss = criterion(outputs, batch_features)
        
          # compute accumulated gradients
          train_loss.backward()
        
          # perform parameter update based on current gradients
          optimizer.step()
        
          # add the mini-batch training loss to epoch loss
          loss += train_loss.item()
    
      # compute the epoch training loss
      loss = loss / len(train_loader)
    
      #  display the epoch training loss
      print("training ::: epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, num_epochs, loss))

def missing_method(test_data, train_data, num_embeddings) :
    
    # test data
    test_data = test_data.copy()
    test_rows, test_cols = test_data.shape
    test_cols -= num_embeddings

    # train data
    train_data = train_data.copy()
    train_rows, train_cols = train_data.shape
    train_cols -= num_embeddings

    # missingness threshold
    t = 0.2

    # uniform random vector, missing values where v<=t
    # test data corruption
    # embedding columns do not have any missing values
    v = np.random.uniform(size=(test_rows, test_cols))
    embeddings_mask = np.zeros((test_rows, num_embeddings), dtype=bool)
    mask = (v<=t)
    mask = np.c_[mask, embeddings_mask]
    # test_data[mask] = np.NAN

    # train data corruption - this is used for training MultipleImputer for imputing mean/median values in the dataset
    v_train = np.random.uniform(size=(train_rows, train_cols))
    embeddings_mask = np.zeros((train_rows, num_embeddings), dtype=bool)
    mask_train = (v_train <= t)
    mask_train = np.c_[mask_train, embeddings_mask]
    # train_data[mask_train] = np.NAN
        
    return test_data, train_data, mask

# uses a multiple imputer to fill the NaN values in the test dataset, calculates mean/median by column
# returns pandas DataFrame

def impute_traditional_all_foods(train, test, method):
  imputer = MultipleImputer(1, strategy=method, return_list=True)
  imputer.fit(train)
  data = imputer.transform(test)
  return data[0][1]

def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total
    return accuracy

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Data preparation
num_epochs = 5
test_size = 0.2
use_cuda = False
batch_size  = 1
num_embeddings = 5

data_pth = '../../dataset/magic/magic04.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
df_data.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
train_col = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
df_data['class'] = df_data['class'].replace({'g':0, 'h':1})
data = df_data

missing_length = 0.2

for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

#indices = test_data['class']
train_data = data[train_col] 
test_data = data[train_col]


train_data = train_data.to_numpy()
#test_data.fillna(0, inplace=True) 
test_data = test_data.to_numpy()
data = df_data.values
rows, cols = data.shape

cols -= 1

y_test = test_data.copy()

missed_data, missed_data_train, mask = missing_method(test_data, train_data, num_embeddings)
print(" === missed_data ====" , missed_data)
print(" === missed_data_train ====" , missed_data_train)
print(" === mask ====" , mask)


num_iterations = 10
imputers = {}
accuracy_list = []
rmse_list = []
results = []

for iteration in range(num_iterations):
    train_data, test_data = train_test_split(data, test_size=test_size)
    
    # train data
    train_data = pd.DataFrame(train_data, columns=columns)
    
    for col in train_col:
        imputer = MultipleImputer(1, strategy='median', return_list=True)
        imputer.fit(train_data)
        imputed_data_list = imputer.transform(train_data)
        imputed_data = imputed_data_list[0][1]

    
    train_imputed_df = pd.DataFrame(imputed_data)
    
    # test data
    test_data = pd.DataFrame(test_data, columns=columns)
    
    for col in train_col:
        imputer = MultipleImputer(1, strategy='median', return_list=True)
        imputer.fit(test_data)
        imputed_data_list = imputer.transform(test_data)
        imputed_data = imputed_data_list[0][1]

    test_imputed_df = pd.DataFrame(imputed_data)

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_data_scaler = scaler.transform(train_imputed_df)
    test_data_scaler = scaler.transform(test_imputed_df)
   

    train_data_noEmbeddings = pd.DataFrame(train_data_scaler, columns = columns)
    train_data_noEmbeddings = np.array(train_data_noEmbeddings.iloc[:, :-num_embeddings]) # df = df.iloc[: , :-1]
    print("=== train_data_noEmbeddings ===", train_data_noEmbeddings)
    test_data_noEmbeddings = pd.DataFrame(test_data_scaler, columns = columns)
    test_data_noEmbeddings = np.array(test_data_noEmbeddings.iloc[:, :-num_embeddings])
    print("=== test_data_noEmbeddings ===", test_data_noEmbeddings)    
    

    missed_data = test_data_noEmbeddings
    missed_data = torch.from_numpy(missed_data).float()
    print("=== missed_data ===", missed_data)
    train_data = torch.from_numpy(train_data_noEmbeddings).float()
    train_loader = torch.utils.data.DataLoader(dataset=train_data_noEmbeddings,
                                           batch_size=batch_size,
                                           shuffle=True)
    missed_data_noEm = missed_data

    # model = Autoencoder(dim=cols-num_embeddings).to(device)
    model = Autoencoder(dim=6).to(device)
    # print("== model ===", model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    training(num_epochs, model, train_loader, criterion, optimizer)

    model.eval()
    rmse_sum = 0

    filled_data = model(missed_data_noEm)
    filled_data = filled_data.cpu().detach().numpy()

    # accuracy
    y_true = np.round(missed_data_noEm.numpy())  # Actual labels
    y_pred = np.round(filled_data)  # Predicted labels

    accuracy = calculate_accuracy(y_true, y_pred)
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")
    accuracy_list.append(accuracy)

    # RMSE 계산 및 저장
    rmse = calculate_rmse(missed_data_noEm.numpy(), filled_data)
    rmse_list.append(rmse)
    print("==========================================")
    print(str(iteration+1)+"th RMSE === : ", rmse)
    print("==========================================")
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)

    
print("=== RMSE result : {:.4f} ± {:.4f}".format(rmse_mean,rmse_std))
    
print("Mean Accuracy: {:.2f}".format(accuracy_mean))
print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(rmse_mean,rmse_std))
print("==========================================")
