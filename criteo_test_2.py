#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import IPNN, xDeepFM

data = pd.read_csv('data/criteo_sample.txt', sep='\t')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
features = data.columns[1:].tolist()

data[sparse_features] = data[sparse_features].fillna(
    '-1',
)
data[dense_features] = data[dense_features].fillna(
    0,
)
target = ['label']
data[target] = data[target].astype(float)

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])


train, test = train_test_split(data, test_size=0.2, random_state=2020)

train_tensor_data = Data.TensorDataset(
    torch.from_numpy(train.iloc[:, 1:].values), torch.from_numpy(train['label'].values)
)
test_tensor_data = Data.TensorDataset(
    torch.from_numpy(test.iloc[:, 1:].values), torch.from_numpy(test['label'].values)
)
train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=16)
test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=1000)
y_test = test[target].values

device = torch.device('cuda:0')

model = IPNN(data.iloc[:, 1:]).double().to(device)
optimizer = torch.optim.Adagrad(model.parameters())
loss = nn.BCEWithLogitsLoss(reduction='sum')

num_epoch = 5
log_interval = 1e3
for epoch in range(1, num_epoch + 1):
    model.train()
    with tqdm(enumerate(train_loader)) as t:
        for i, (x_train, y_train) in t:
            optimizer.zero_grad()
            x = x_train.to(device)
            y = y_train.to(device)

            y_hat = model(x).squeeze()
            l = loss(y_hat, y.squeeze())
            l += torch.sum(
                1e-5
                * torch.square(
                    torch.cat([i.weight for i in model.embedding_dict.values()])
                )
            )
            if model.linear:
                l += torch.sum(
                    1e-5
                    * torch.square(
                        torch.cat(
                            [i.weight for i in model.linear_embedding_dict.values()]
                        )
                    )
                )
            l.backward()
            optimizer.step()

            if (i + 1) % log_interval == 0 or i == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        (i + 1) * len(x),
                        len(train_loader.dataset),
                        100.0 * (i + 1) / len(train_loader),
                        l.item(),
                    )
                )

    model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         y_hat = model(x)
    #         test_loss += F.binary_cross_entropy_with_logits(
    #             y_hat.squeeze(), y.squeeze(), reduction='sum'
    #         ).item()
    #         pred = y_hat.argmax(
    #             dim=1, keepdim=True
    #         )  # get the index of the max log-probability
    #         correct += pred.eq(y.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    y_hat = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y_hat.append(model(x).squeeze())

        y_hat = torch.cat(y_hat).cpu().numpy()
        test_loss = log_loss(y_test, y_hat)
        auc = roc_auc_score(y_test, y_hat)

    print('\nTest set: Log loss: {:.4f}, AUC: {:.4f}%)\n'.format(test_loss, auc))
