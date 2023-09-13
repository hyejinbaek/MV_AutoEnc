import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class DynamicImputationNN(nn.Module):
    
    def __init__(self, dim_x, dim_y, seed, num_hidden=50, num_layers=1, lr=1e-3, batch_size=32, max_epochs=500):
        super(DynamicImputationNN, self).__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        
        # Reset the random seed
        torch.manual_seed(seed)
        
        self.logits, self.pred = self.prediction()
        
        self.imputer = IterativeImputer(sample_posterior=True, random_state=self.seed)
        
    def prediction(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(self.dim_x, self.num_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.num_hidden, self.dim_y))
        if self.dim_y == 1:
            layers.append(nn.Sigmoid())
        elif self.dim_y > 2:
            layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers), None  # We won't use pred in PyTorch
    
    def forward(self, x):
        return self.logits(x)
    
    def train_with_dynamic_imputation(self, x_trnval, y_trnval, save_path, num_mi, m, tau, early_stopping=True):
        # Numpy 배열을 Tensor로 변환
        x_trnval = torch.tensor(x_trnval, dtype=torch.float32)
        y_trnval = torch.tensor(y_trnval, dtype=torch.float32)
        
        self.imputer.fit(x_trnval)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trnval, y_trnval, random_state=self.seed, test_size=0.2)
        
        x_val_imputed_list = [self.imputer.transform(x_val) for _ in range(num_mi)]
        x_val_imputed = torch.tensor(np.mean(x_val_imputed_list, axis=0), dtype=torch.float32)
        
        n_batch = int(len(x_trn) / self.batch_size)
        
        if self.dim_y == 1:
            loss_fn = nn.BCEWithLogitsLoss()
        elif self.dim_y > 2:
            loss_fn = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        print('::::: training')
        
        val_log = np.zeros(self.max_epochs)
        
        imputed_list = []
        
        for epoch in range(self.max_epochs):
            
            x_trn_imputed = self.imputer.transform(x_trn)
            imputed_list.append(x_trn_imputed)
                    
            [x_trn_input, y_trn_input] = self._permutation([x_trn_imputed, y_trn])
  
            for i in range(n_batch):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size
                x_batch = torch.tensor(x_trn_input[start_:end_], dtype=torch.float32)
                y_batch = torch.tensor(y_trn_input[start_:end_], dtype=torch.float32)
                optimizer.zero_grad()
                logits = self(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
            
            val_x = x_val_imputed
            val_y = y_val
            val_loss = loss_fn(self(val_x), val_y).item()
            val_log[epoch] = val_loss
            print('epoch: %d, val_loss: %f, BEST: %f'%(epoch+1, val_loss, np.min(val_log[:epoch+1])))
            
            if early_stopping:
                if np.min(val_log[:epoch+1]) == val_loss:
                    torch.save(self.state_dict(), save_path)

                if epoch > 20 and np.min(val_log[epoch-20:epoch+1]) > np.min(val_log[:epoch-20]):
                    self.load_state_dict(torch.load(save_path))
                    break
    
            # imputation stopping rule
            if epoch >= m - 1:
                print(" === stopping ===")
                missing_mask = np.isnan(x_trn.cpu().numpy()).astype(int)
                missing_num = np.sum(missing_mask)
                
                missing_idx = np.where(missing_mask == 1)
                element_wise_missing_idx_list = [[missing_idx[0][i], missing_idx[1][i]] for i in range(missing_num)]
                
                recent_mean = torch.tensor(np.mean(imputed_list[epoch-(m-1):], axis=0), dtype=torch.float32)
                recent_var = torch.tensor(np.var(imputed_list[epoch-(m-1):], axis=0, ddof=1), dtype=torch.float32)
                
                for idx in element_wise_missing_idx_list:
                    if recent_var[idx[0], idx[1]] < tau:
                        x_trn[idx[0], idx[1]] = recent_mean[idx[0], idx[1]]
            
        
    def get_accuracy(self, x_tst, y_tst):
                
        if self.dim_y == 1:
            pred_Y = torch.sigmoid(self(torch.tensor(x_tst, dtype=torch.float32)))
            pred_Y = (pred_Y > 0.5).float()
            acc = accuracy_score(y_tst, pred_Y.detach().numpy())
        else:
            with torch.no_grad():
                y_tst_hat = torch.softmax(self(x_tst), dim=1).numpy()
            y_tst_hat = np.argmax(y_tst_hat, axis=1)
            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)
        
        return acc
   
    def get_auroc(self, x_tst, y_tst):
        with torch.no_grad():
            y_tst_hat = torch.softmax(self(x_tst), dim=1).numpy()
        if self.dim_y == 1:
            auroc = roc_auc_score(y_tst, y_tst_hat)
        else:
            auroc = roc_auc_score(y_tst, y_tst_hat, average='macro', multi_class='ovr')
        return auroc
    
    def _permutation(self, set):
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
        return set
