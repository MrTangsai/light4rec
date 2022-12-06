'''
@File         :basedata.py
@Description  :data base class 
@Time         :2022/11/14 23:30:21
@Author       :tangs
@Version      :1.0
'''

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as Data
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, random_split
from .features import FeatureMap


class BaseRecData(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        batch_size=16,
        var_batch_size=1000,
    ) -> None:
        super().__init__()
        self.data_cfg = yaml.load(open(data_cfg, 'r'), yaml.FullLoader)
        self._load_data()

        self.batch_size = batch_size
        self.var_batch_size = var_batch_size
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        pass

    def _load_data(self):
        data_dir = self.data_cfg.get('data_dir')
        if isinstance(data_dir, str):
            if data_dir.endswith('.txt') or data_dir.endswith('.csv'):
                data = pd.read_csv(
                    data_dir, sep=self.data_cfg.get('sep'), engine='python'
                )
            else:
                raise TypeError('data file not exists')
        elif isinstance(data_dir, list):
            for d in data_dir:
                # todo
                if 'train' in d and (d.endswith('.txt') or d.endswith('.csv')):
                    data = pd.read_csv(d, sep=self.data_cfg.get('sep'), nrows=4e5)
                    break

        self.columns = data.columns.tolist()
        label = self.data_cfg['label_col']
        self.label_name = label['name']
        self.label = data[self.label_name].values
        if label['dtype'] == 'float':
            self.label = self.label.astype(float)
        self.columns.remove(self.label_name)

        features_attr = dict()
        self.encoders = OrderedDict()
        self.array = []

        index, dense, sparse = 0, 0, 0
        for cols in self.data_cfg.get('feature_cols'):
            if cols['type'] == 'dense':
                for col in cols['name']:
                    features_attr[col] = {}
                    if cols.get('fillna'):
                        data[col] = self._fillna(data[col], cols['fillna'], 0)
                    if cols.get('encoder'):
                        out, encoder = self._encode(data[col], cols['encoder'])
                        self.encoders[col] = encoder
                    else:
                        out = data[col].values.reshape(-1, 1)
                    self.array.append(out)
                    features_attr[col]['type'] = 'dense'
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    dense += 1
            if cols['type'] == 'categorical':
                for col in cols['name']:
                    features_attr[col] = {}
                    if cols.get('fillna'):
                        data[col] = self._fillna(data[col], cols['fillna'], '-1')
                    if cols.get('encoder'):
                        out, encoder = self._encode(data[col], cols['encoder'])
                        self.encoders[col] = encoder
                        features_attr[col]['size'] = len(encoder.classes_)
                    else:
                        out = data[col].values.reshape(-1, 1)
                    self.array.append(out)
                    features_attr[col]['type'] = 'categorical'
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    sparse += 1

        self.featuremap = FeatureMap(
            num_fields=dense + sparse,
            dense_features=dense,
            sparse_features=sparse,
            num_features=sum(
                [i['size'] for i in features_attr.values() if i.get('size')]
            ),
            input_length=index,
            features_attr=features_attr,
        )

        self.array = np.hstack(self.array)

        # self.sparse_features = {
        #     j: i for i, j in enumerate(self.columns) if j in self.sparse_features
        # }
        # self.dense_features = {
        #     j: i for i, j in enumerate(self.columns) if j in self.dense_features
        # }

    def _fillna(self, raw, type, constant):
        if type == 'constant':
            return raw.fillna(constant)

    def _encode(self, raw, type):
        if type == 'LabelEncoder':
            m = LabelEncoder()
            raw = m.fit_transform(raw.values).reshape(-1, 1)
        elif type == 'MinMaxScaler':
            m = MinMaxScaler(feature_range=(0, 1))
            raw = m.fit_transform(raw.values.reshape(-1, 1))

        return raw, m

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data = Data.TensorDataset(
                torch.from_numpy(self.array),
                torch.from_numpy(self.label),
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
