'''
@File         :basemodel.py
@Description  :
@Time         :2022/11/29 10:03:14
@Author       :tangs
@Version      :1.0
'''


import os
import warnings

import yaml
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchmetrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        data_cfg,
        batch_size=16,
        var_batch_size=1000,
        embedding_dim=4,
        linear=False,
    ) -> None:
        super(BaseModel, self).__init__()
        self.data_cfg = yaml.load(open(data_cfg, 'r'), yaml.FullLoader)
        self.batch_size = batch_size
        self.var_batch_size = var_batch_size

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.auc_val = torchmetrics.AUROC()
        self.auc_test = torchmetrics.AUROC()

        self._load_data()

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

        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _load_data(self):
        data_dir = self.data_cfg.get('data_dir')
        if isinstance(data_dir, str):
            if data_dir.endswith('.txt') or data_dir.endswith('.csv'):
                data = pd.read_csv(data_dir, sep=self.data_cfg.get('sep'))
            else:
                raise TypeError('data file not exists')
        elif isinstance(data_dir, list):
            for d in data_dir:
                # todo
                if 'train' in d and (d.endswith('.txt') or d.endswith('.csv')):
                    data = pd.read_csv(d, sep=self.data_cfg.get('sep'), nrows=4e5)
                    break

        self.sparse_features = []
        self.dense_features = []

        for col in self.data_cfg.get('feature_cols'):
            if col['type'] == 'dense':
                self.dense_features += col['name']
                if col.get('fillna'):
                    data[self.dense_features] = self._fillna(
                        data[self.dense_features], col['fillna'], 0
                    )
                if col.get('encoder'):
                    data[self.dense_features] = self._encode(
                        data[self.dense_features], col['encoder']
                    )
            if col['type'] == 'categorical':
                self.sparse_features += col['name']
                if col.get('fillna'):
                    data[self.sparse_features] = self._fillna(
                        data[self.sparse_features], col['fillna'], '-1'
                    )
                if col.get('encoder'):
                    data[self.sparse_features] = self._encode(
                        data[self.sparse_features], col['encoder']
                    )

        self.columns = data.columns.tolist()
        label = self.data_cfg['label_col']
        self.label_name = label['name']
        if label['dtype'] == 'float':
            data[self.label_name] = data[self.label_name].astype(float)

        self.columns.remove(self.label_name)
        self.sparse_features = {
            j: i for i, j in enumerate(self.columns) if j in self.sparse_features
        }
        self.dense_features = {
            j: i for i, j in enumerate(self.columns) if j in self.dense_features
        }
        self.data = data

    def _fillna(self, raw, type, constant):
        if type == 'constant':
            return raw.fillna(constant)

    def _encode(self, raw, type):
        if type == 'LabelEncoder':
            for feat in raw.columns:
                lbe = LabelEncoder()
                raw[feat] = lbe.fit_transform(raw[feat])
        if type == 'MinMaxScaler':
            mms = MinMaxScaler(feature_range=(0, 1))
            raw = mms.fit_transform(raw)
        return raw

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


class SquenceModel(BaseModel):
    def __init__(
        self,
        data_cfg,
        din_target_field=None,
        din_sequence_field=None,
        batch_size=16,
        var_batch_size=1000,
        embedding_dim=4,
        linear=False,
    ) -> None:
        super().__init__(
            data_cfg,
            batch_size,
            var_batch_size,
            embedding_dim,
            linear,
        )
        if not isinstance(din_target_field, list):
            din_target_field = [din_target_field]
        self.din_target_field = {
            j: i for i, j in enumerate(self.columns) if j in self.din_target_field
        }
        if not isinstance(din_sequence_field, list):
            din_sequence_field = [din_sequence_field]
        for seq in self.din_sequence_field:
            seqdata = self.data[seq].str.split('^', expand=True)
            seqdata.columns = [seq + '_' + str(i) for i in seqdata.columns]
            self.data = pd.concat([self.data, seqdata], axis=1)
        self.din_sequence_field = {
            k: [j for j, i in enumerate(self.columns) if i.startswith(k)]
            for k in self.din_sequence_field
        }
        if len(self.din_target_field) != len(self.din_sequence_field):
            raise AttributeError(
                'din_target_field length is not equal to din_sequence_field length'
            )
        if self.din_sequence_field:
            # share embedding
            self.embedding_dict.update(
                nn.ModuleDict(
                    {
                        feat: self.embedding_dict[self.din_target_field[index]]
                        for index, feat in enumerate(self.din_sequence_field)
                    }
                )
            )
