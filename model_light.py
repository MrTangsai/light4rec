'''
@File         :model_light.py
@Description  :model class for recommedation used by pytorch-lightning
@Time         :2022/11/13 17:51:54
@Author       :tangs
@Version      :1.0
'''

import os
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchmetrics
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseRecData(pl.LightningDataModule):
    """Example of LightningDataModule for recommedation dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(self, data_dir, label_name='name', batch_size=16, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.label_name = label_name
        self.batch_size = batch_size
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _load_data(self):
        data = pd.read_csv(self.data_dir, **self.kwargs)

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

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        data_dir,
        label_name='name',
        batch_size=16,
        var_batch_size=1000,
        embedding_dim=4,
        linear=False,
        **kwargs
    ) -> None:
        super(BaseModel, self).__init__()
        self.data_dir = data_dir
        self.label_name = label_name
        self.batch_size = batch_size
        self.var_batch_size = var_batch_size
        self.kwargs = kwargs
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.auc_val = torchmetrics.AUROC()
        self.auc_test = torchmetrics.AUROC()
        (
            self.columns,
            self.sparse_features,
            self.dense_features,
            self.data,
        ) = self._load_data()
        self.sparse_features = {
            j: i for i, j in enumerate(self.columns) if j.startswith('C')
        }
        self.dense_features = {
            j: i for i, j in enumerate(self.columns) if j.startswith('I')
        }

        self.embedding_dict = nn.ModuleDict(
            {
                feat: nn.Embedding(
                    num_embeddings=int(self.data[feat].max() + 1),
                    embedding_dim=embedding_dim,
                )
                for feat in self.sparse_features.keys()
            }
        )
        self.linear = linear
        if self.linear:
            self.linear_embedding_dict = nn.ModuleDict(
                {
                    feat: nn.Embedding(
                        num_embeddings=int(self.data[feat].max() + 1),
                        embedding_dim=1,
                    )
                    for feat in self.sparse_features.keys()
                }
            )
            self.linear_dense = nn.Linear(len(self.dense_features), 1, bias=False)

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _load_data(self):
        sep = self.kwargs.get('sep')
        data = pd.read_csv(self.data_dir, sep=sep)

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

        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        return features, sparse_features, dense_features, data

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data = Data.TensorDataset(
                torch.from_numpy(self.data.drop(columns=[self.label_name]).values),
                torch.from_numpy(self.data[self.label_name].values),
            )

            length = [int(i * len(data)) for i in [0.7, 0.2, 0.1]]
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=data, lengths=length
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.var_batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.var_batch_size, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters())
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        l = self.loss(y_hat, y.squeeze())
        # self.log('val_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {'val_loss': l, 'val_auc': self.auc_val(y_hat, y.squeeze().long())},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        l = self.loss(y_hat, y.squeeze())
        # self.log('test_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {'test_loss': l, 'test_auc': self.auc_test(y_hat, y.squeeze().long())},
            # 'test_auc',
            # self.auc_test(y_hat, y.squeeze().long()),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )


class CIN(nn.Module):
    def __init__(self, init_size, cin_layer_size=(256, 128)) -> None:
        super(CIN, self).__init__()
        self.cin_layer_size = cin_layer_size
        self.cin_convs = nn.ModuleList()
        self.cin_convs.append(
            nn.Conv1d(init_size**2, self.cin_layer_size[0], kernel_size=1)
        )
        for index, size in enumerate(self.cin_layer_size[1:]):
            self.cin_convs.append(
                nn.Conv1d(
                    int(init_size * self.cin_layer_size[index] / 2),
                    int(size),
                    kernel_size=1,
                )
            )

    def forward(self, x):
        cross_layer, hidden_layer = [x], [x]
        batch_size, _, dim = x.size()

        for i, size in enumerate(self.cin_layer_size):
            hidden_output = self.cin_convs[i](
                torch.einsum('abc,adc->abdc', cross_layer[0], cross_layer[-1]).view(
                    batch_size, -1, dim
                )
            )
            hidden_output = F.relu(hidden_output)
            if i == len(self.cin_layer_size) - 1:
                hidden_layer.append(hidden_output)
            else:
                h1, h2 = torch.split(hidden_output, size // 2, 1)
                cross_layer.append(h1)
                hidden_layer.append(h2)

        output = torch.sum(torch.cat(hidden_layer[1:], 1), -1)
        return output


class xDeepFM(BaseModel):
    def __init__(
        self,
        data_dir,
        label_name='label',
        cin_layer_size=(256, 128),
        dnn_hidden_size=(256, 256),
        embedding_dim=4,
        dnn_dropout=0,
    ) -> None:
        super().__init__(data_dir, label_name, embedding_dim, linear=True)
        self.cin_layer_size = cin_layer_size
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_dropout = dnn_dropout
        self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]

        self.cin = CIN(len(self.sparse_features), cin_layer_size)
        self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False)

        dnn_input = int(
            len(self.dense_features) + embedding_dim * len(self.sparse_features)
        )
        self.dnn_hidden_size = (dnn_input,) + self.dnn_hidden_size
        dnn_hidden_size_tuple = list(
            zip(self.dnn_hidden_size[:-1], self.dnn_hidden_size[1:])
        )
        self.dnn = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(*size), nn.ReLU(), nn.Dropout(self.dnn_dropout))
                for size in dnn_hidden_size_tuple
            ]
        )
        self.dnn_linear = nn.Linear(self.dnn_hidden_size[-1], 1, bias=False)
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, x):
        dense_value_list = [
            x[:, index].view(-1, 1) for index in self.dense_features.values()
        ]
        sparse_linear_embedding_list = [
            self.linear_embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.sparse_features.items()
        ]

        sparse_embedding_cat = torch.cat(sparse_linear_embedding_list, dim=-1)
        sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
        dense_value_logit = self.linear_dense(torch.cat(dense_value_list, dim=-1))
        linear_logit = sparse_feat_logit + dense_value_logit

        sparse_embedding_list = [
            self.embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.sparse_features.items()
        ]
        cin_input = torch.cat(sparse_embedding_list, dim=1)
        cin_output = self.cin(cin_input)
        cin_logit = self.cin_linear(cin_output)

        dnn_input = torch.cat(
            [
                torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1),
                torch.cat(dense_value_list, dim=1),
            ],
            dim=1,
        )
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        final_logit = linear_logit + dnn_logit + cin_logit

        return final_logit

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x).squeeze()
        l = self.loss(y_hat, y.squeeze())
        l += torch.sum(
            1e-5
            * torch.square(torch.cat([i.weight for i in self.embedding_dict.values()]))
        )
        if self.linear:
            l += torch.sum(
                1e-5
                * torch.square(
                    torch.cat([i.weight for i in self.linear_embedding_dict.values()])
                )
            )
        self.log(
            "train_loss", l, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return l


if __name__ == '__main__':
    from pytorch_lightning.callbacks import RichProgressBar
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
    from pytorch_lightning.callbacks import RichModelSummary
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    # create your own theme!
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    device = torch.device('cuda:0')

    model = xDeepFM(
        'rec/data/criteo_sample.txt',
    ).double()

    # train model
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=10,
        # limit_train_batches=10,
        # limit_val_batches=10,
        callbacks=[
            progress_bar,
            RichModelSummary(),
            EarlyStopping(monitor="val_loss", mode="min"),
        ],
    )
    trainer.fit(model=model)
    trainer.test(model)
