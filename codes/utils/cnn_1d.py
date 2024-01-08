import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class AMLDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        if y is not None:
            self.y = torch.from_numpy(y).float()
        else:
            self.y = None

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        else:
            return self.X[index]

    def __len__(self):
        return len(self.X)


class CNN1DClassifier(nn.Module):
    def __init__(self, device_name, num_features, num_targets, **kwargs):
        super(CNN1DClassifier, self).__init__()
        hidden_size = kwargs.get('hidden_size', 512)
        self.criterion = kwargs.get('criterion', nn.BCEWithLogitsLoss)
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam)
        self.scheduler = kwargs.get('scheduler', torch.optim.lr_scheduler.OneCycleLR)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.batch_size = kwargs.get('batch_size', 128)
        self.epochs = 100
        self.patience = 10

        self.num_targets = num_targets
        if device_name is None:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)

        cha_1 = 8
        cha_2 = 16
        cha_3 = 16

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        # TODO: Can't adtapt to different input size, dataloader must set drop_last=True
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                          self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        return x
    
    def fit(self, X_train, y_train, path=None, max_evals=56, eval_set=None, feature_name=None):
        self.to(self.device)
        # change y to one-hot
        y_train = np.eye(self.num_targets)[y_train]
        if eval_set is not None:
            x_val, y_val = eval_set[0][0], np.eye(self.num_targets)[eval_set[0][1]]
            best_val_loss = np.inf
            no_improvement_count = 0
        train_loader = DataLoader(AMLDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        if eval_set is not None:
            val_loader = DataLoader(AMLDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False)
        criterion = self.criterion()
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = self.scheduler(optimizer, max_lr=self.learning_rate, total_steps=self.epochs)
        for epoch in range(self.epochs):
            self.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if eval_set is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    for i, (inputs, labels) in enumerate(val_loader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        # keep dim of each element
                        max_indices = torch.argmax(outputs, dim=1)
                        one_hot = torch.zeros_like(outputs)
                        one_hot[torch.arange(outputs.shape[0]), max_indices] = 1
                        preds = one_hot
                        
                        val_acc += (preds == labels).sum().item()
                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader.dataset)
                    # print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, self.epochs, val_loss, val_acc))

                    # early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_weights = copy.deepcopy(self.state_dict())
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count == self.patience:
                            print('Early stopping at epoch {}'.format(epoch+1))
                            self.load_state_dict(best_model_weights)
                            return
                            
            scheduler.step(val_loss)
        print('Finished Training')

    def save_model(self, path, feature_name, max_evals=56):
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(max_evals):
            if not os.path.exists(path+f'/cnn_1d.{feature_name}.{i}.pth'):
                torch.save(self.state_dict(), path+f'/cnn_1d.{feature_name}.{i}.pth')
                break

    def predict(self, X_test):
        self.to(self.device)
        test_loader = DataLoader(AMLDataset(X_test), batch_size=self.batch_size, shuffle=False)
        self.eval()
        with torch.no_grad():
            preds = []
            for i, inputs in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                max_indices = torch.argmax(outputs, dim=1)
                preds.append(max_indices.cpu().numpy())
            preds = np.concatenate(preds)
            return preds
        
    def predict_proba(self, X_test):
        self.to(self.device)
        test_loader = DataLoader(AMLDataset(X_test), batch_size=self.batch_size, shuffle=False)
        self.eval()
        with torch.no_grad():
            preds = []
            for i, inputs in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = self(inputs)

                outputs = F.softmax(outputs, dim=1)
                preds.append(outputs.cpu().numpy())
            preds = np.concatenate(preds)
            return preds