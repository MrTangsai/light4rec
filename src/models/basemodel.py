'''
@File         :basemodel.py
@Description  :
@Time         :2022/11/29 10:03:14
@Author       :tangs
@Version      :1.0
'''


import os
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


warnings.filterwarnings("ignore")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        featuremap,
        embedding_dim=4,
        linear=False,
    ) -> None:
        super(BaseModel, self).__init__()
        self.featuremap = featuremap
        self.auc_val = torchmetrics.AUROC()
        self.auc_test = torchmetrics.AUROC()

        self.embedding_dict = nn.ModuleDict(
            {
                feat: nn.Embedding(
                    num_embeddings=int(m['size']),
                    embedding_dim=embedding_dim,
                )
                for feat, m in self.features_attr.items()
                if m['type'] == 'categorical'
            }
        )
        self.linear = linear
        if self.linear:
            self.linear_embedding_dict = nn.ModuleDict(
                {
                    feat: nn.Embedding(
                        num_embeddings=int(m['size']),
                        embedding_dim=1,
                    )
                    for feat, m in self.features_attr.items()
                    if m['type'] == 'categorical'
                }
            )
            if self.featuremap.dense_features > 0:
                self.linear_dense = nn.Linear(
                    self.featuremap.dense_features, 1, bias=False
                )

        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self.featuremap._fields:
                return self.featuremap.__getattribute__(name)
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

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
        feature_map,
        din_target_field=None,
        din_sequence_field=None,
        embedding_dim=4,
        linear=False,
    ) -> None:
        super().__init__(
            feature_map,
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
