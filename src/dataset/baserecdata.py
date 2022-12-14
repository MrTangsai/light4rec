'''
@File         :basedata.py
@Description  :data base class 
@Time         :2022/11/14 23:30:21
@Author       :tangs
@Version      :1.0
'''

import threading
import time
from collections import OrderedDict
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as Data
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .encoders import Tokenizer
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
        self.encoders = OrderedDict()
        self._extract_featuremap()
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

    def _extract_featuremap(self):
        data = self._read_data()
        self._init_encoders()
        data = data if 'chunksize' in data.__dict__ else [data]
        features_attr = dict()
        index, dense, sparse, sequence = 0, 0, 0, 0
        for cols in self.data_cfg.get('feature_cols'):
            cols_list = (
                cols['name'] if isinstance(cols['name'], list) else [cols['name']]
            )
            for col in cols_list:
                features_attr[col] = {}
                if cols['type'] == 'dense':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    dense += 1
                elif cols['type'] == 'categorical':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    sparse += 1
                elif cols['type'] == 'sequence':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['index'] = [index, index + cols['max_len']]
                    features_attr[col]['share_embedding'] = cols.get('share_embedding')
                    index += cols['max_len']
                    sequence += 1
        for d in data:
            self._fit_encoders(d)
        for name, encoder in self.encoders.items():
            if isinstance(encoder, Tokenizer):
                encoder.build_vocab()
                features_attr[name]['size'] = self.encoders[name].vocab_size
                if encoder.share_embedding:
                    encoder.vocab = self.encoders[encoder.share_embedding].vocab
                    encoder.vocab_size = self.encoders[
                        encoder.share_embedding
                    ].vocab_size
        self.featuremap = FeatureMap(
            num_fields=dense + sparse + sequence,
            dense_features=dense,
            sparse_features=sparse,
            sequence_features=sequence,
            num_features=sum(
                [i['size'] for i in features_attr.values() if i.get('size')]
            ),
            input_length=index,
            features_attr=features_attr,
        )

    def _init_encoders(self):
        for cols in self.data_cfg.get('feature_cols'):
            if cols.get('encoder'):
                cols_list = (
                    cols['name'] if isinstance(cols['name'], list) else [cols['name']]
                )
                for col in cols_list:
                    if cols['type'] == 'dense':
                        self.encoders[col] = MinMaxScaler(feature_range=(0, 1))
                    elif cols['type'] == 'categorical':
                        self.encoders[col] = Tokenizer(na_value='-1')
                    elif cols['type'] == 'sequence':
                        self.encoders[col] = (
                            Tokenizer(
                                na_value='-1', share_embedding=cols['share_embedding']
                            )
                            if cols.get('share_embedding')
                            else Tokenizer(na_value='-1')
                        )
                    else:
                        raise NotImplementedError

    def _fit_encoders(self, data):
        print_thread('start')
        for cols in self.data_cfg.get('feature_cols'):
            cols_list = (
                cols['name'] if isinstance(cols['name'], list) else [cols['name']]
            )
            for col in cols_list:
                if cols['type'] == 'dense':
                    if cols.get('encoder'):
                        raw = data[col].fillna(0)
                        self.encoders[col].partial_fit(raw.values.reshape(-1, 1))
                elif cols['type'] == 'categorical':
                    if cols.get('encoder'):
                        raw = data[col].fillna('-1')
                        self.encoders[col].partial_fit(raw.values)
                elif cols['type'] == 'sequence':
                    if cols.get('encoder') and not cols.get('share_embedding'):
                        raw = data[col]
                        self.encoders[col].partial_fit(
                            raw.str.split(cols['splitter'], expand=True)
                            .fillna('-1')
                            .values.flatten()
                        )
                else:
                    raise NotImplementedError
        print_thread('end')

    def _fit_array(self):
        data = self._read_data()

        self.array = []
        self.label = data[self.label_name].values
        if self.data_cfg['label_col'].get('dtype') == 'float':
            self.label = self.label.astype(float)

        for cols in self.data_cfg.get('feature_cols'):
            for col in cols['name']:
                if cols['type'] == 'dense':
                    out = data[col].fillna(0).values.reshape(-1, 1)
                    if col in self.encoders:
                        out = self.encoders[col].transform(out)
                elif cols['type'] == 'categorical':
                    if cols.get('encoder'):
                        out = data[col].fillna('-1')
                        if col in self.encoders:
                            out = self.encoders[col].encode_category(out)
                else:
                    out = data[col].values.reshape(-1, 1)
                self.array.append(out)

        self.array = np.hstack(self.array)

    def _read_data(self):
        train_dir = Path(self.data_cfg.get('data_dir')).joinpath(
            self.data_cfg.get('train_data')
        )
        if not train_dir.exists():
            raise FileNotFoundError(f"No such file or directory: '{train_dir}'")
        self.sep = self.data_cfg.get('sep')
        self.chunksize = self.data_cfg.get('chunksize')
        self.label_name = self.data_cfg['label_col']['name']
        if train_dir.suffix in ['.csv', '.txt']:
            self.columns = pd.read_csv(
                train_dir, sep=self.sep, nrows=0
            ).columns.to_list()
            self.columns.remove(self.label_name)

            data = eval(self.data_cfg.get('reply', 'pd')).read_csv(
                train_dir,
                sep=self.sep,
                dtype={
                    j: i['dtype']
                    for i in self.data_cfg['feature_cols']
                    for j in (i['name'] if isinstance(i['name'], list) else [i['name']])
                },
                chunksize=self.chunksize,
            )
        else:
            raise NotImplementedError

        return data

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self._fit_array()
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


def print_thread(pos='start'):
    t = threading.currentThread()
    print(
        f'{pos}:\t',
        'ID: ',
        t.ident,
        'name: ',
        t.getName(),
        'time: ',
        time.asctime(time.localtime(time.time())),
        '\n',
    )
