'''
@File         :basedata.py
@Description  :data base class 
@Time         :2022/11/14 23:30:21
@Author       :tangs
@Version      :1.0
'''

import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pytorch_lightning as pl
import torchdata.datapipes as dp
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..decorators import log_filter, logger
from .datapipe import CSVHeader, CSVReader, NPYReader, rand_split_train_val
from .encoders import Tokenizer
from .features import FeatureMap


class TorchRecData(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg,
        fm_path=None,
        batch_size=16,
        var_batch_size=1000,
    ) -> None:
        super().__init__()
        self.data_cfg = yaml.load(open(data_cfg, 'r'), yaml.FullLoader)
        if 'process_dir' in self.data_cfg:
            self.process_dir = Path(self.process_dir)
        else:
            self.process_dir = None
        if fm_path:
            self.load(fm_path)
        else:
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

    @log_filter
    def _extract_featuremap(self):
        columns, dataiter = self._read_data()
        self._init_encoders()
        features_attr = OrderedDict()
        index, dense, sparse, sequence = 0, 0, 0, 0
        for cols in self.feature_cols:
            cols_list = (
                cols['name'] if isinstance(cols['name'], list) else [cols['name']]
            )
            for col in cols_list:
                features_attr[col] = {}
                features_attr[col]['rowindex'] = columns.index(col)
                if cols['type'] == 'dense':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['dtype'] = cols['dtype']
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    dense += 1
                elif cols['type'] == 'categorical':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['dtype'] = cols['dtype']
                    features_attr[col]['index'] = [index, index + 1]
                    index += 1
                    sparse += 1
                elif cols['type'] == 'sequence':
                    features_attr[col]['type'] = cols['type']
                    features_attr[col]['dtype'] = cols['dtype']
                    features_attr[col]['index'] = [index, index + cols['max_len']]
                    features_attr[col]['share_embedding'] = cols.get('share_embedding')
                    index += cols['max_len']
                    sequence += 1
        # label_attr = dict()
        label = self.label_col['name']
        label_attr = {
            'name': label,
            'index': columns.index(label),
            'dtype': self.label_col['dtype'],
        }
        dl = DataLoader(
            dataset=dataiter.sharding_filter(),
            batch_size=self.chunksize,
            num_workers=2,
        )
        for d in dl:
            self._fit_encoders(d, features_attr)
        for name, encoder in self.encoders.items():
            if isinstance(encoder, Tokenizer):
                encoder.build_vocab()
                if encoder.share_embedding:
                    encoder.vocab = self.encoders[encoder.share_embedding].vocab
                    encoder.vocab_size = self.encoders[
                        encoder.share_embedding
                    ].vocab_size
                features_attr[name]['size'] = self.encoders[name].vocab_size
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
            label_attr=label_attr,
        )

    def _init_encoders(self):
        self.encoders = OrderedDict()
        for cols in self.feature_cols:
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
                        self.encoders[col] = Tokenizer(
                            na_value='-1',
                            padding=True,
                            share_embedding=cols.get('share_embedding'),
                            splitter=cols.get('splitter'),
                        )
                        if cols.get('share_embedding'):
                            self.encoders[cols['share_embedding']].padding = True

                    else:
                        raise NotImplementedError

    def _fit_encoders(self, data, attrs):
        for col, attr in attrs.items():
            raw = np.array(data[attr['rowindex']])
            if col in self.encoders:
                if attr['type'] == 'dense':
                    # raw = np.nan_to_num(raw, nan=0)
                    raw[np.where(raw == '')] = 0
                    raw = raw.astype(attr['dtype'])
                    self.encoders[col].partial_fit(raw.reshape(-1, 1))
                elif attr['type'] == 'categorical':
                    raw[np.where(raw == '')] = '-1'
                    raw = raw.astype(attr['dtype'])
                    self.encoders[col].partial_fit(raw)
                elif attr['type'] == 'sequence':
                    if not self.encoders[col].share_embedding:
                        raw[np.where(raw == '')] = '-1'
                        raw = raw.astype(attr['dtype'])
                        self.encoders[col].partial_fit(raw)
                else:
                    raise NotImplementedError

    def _read_data(self, stage='train'):
        if stage == 'train':
            path = Path(self.data_dir).joinpath(self.train_data)
        elif stage == 'test':
            path = Path(self.data_dir).joinpath(self.test_data)
        else:
            raise AttributeError("stage have to be train or test")
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        sep = self.data_cfg.get('sep')
        if path.suffix in ['.csv', '.txt']:
            columns = next(iter(CSVHeader(path, sep=sep)))
            data = CSVReader(path, sep=sep)
        else:
            raise NotImplementedError

        return columns, data

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if not self.process_dir:
                data = self._transform_data_to_iter()
            elif not list(self.process_dir.glob('data*.npy')):
                data = self._transform_data_to_iter(dumps=True)
            else:
                data = NPYReader(self.process_dir).map(lambda x: (x[:-1], x[-1]))
            self.data_train, self.data_test = rand_split_train_val(data.shuffle(), 0.9)
            self.data_train, self.data_val = rand_split_train_val(self.data_train, 0.8)

    @log_filter
    def _transform_data_to_iter(self, dumps=False):
        columns, data_iter = self._read_data()
        self.featuremap.features_attr
        data_iter = data_iter.batch(
            self.data_cfg.get('chunk_size', 1000000)
        ).rows2columnar(column_names=columns)
        data = dp.iter.IterableWrapper([])
        label = dp.iter.IterableWrapper([])
        for num, d in tqdm(enumerate(data_iter)):
            x_set = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures_to_index = {
                    executor.submit(self._transform_col_mp, d, col, fm): index
                    for index, (col, fm) in enumerate(
                        self.featuremap.features_attr.items()
                    )
                    if fm['type'] != 'sequence'
                }
                for future in as_completed(futures_to_index):
                    index = futures_to_index[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(e)
                    else:
                        x_set.append((index, result))
                        x_set.sort(key=lambda x: x[0])
                        x = np.hstack([i[1] for i in x_set]).astype(np.float64)
            y = np.array(
                d[self.featuremap.label_attr['name']],
                dtype=self.featuremap.label_attr['dtype'],
            )
            if dumps:
                np.save(
                    self.process_dir.joinpath(f'data{num+1}.npy'),
                    np.hstack([x, y.reshape(-1, 1)]),
                )
            data = data.concat(dp.iter.IterableWrapper(x))
            label = label.concat(dp.iter.IterableWrapper(y))
            del x, y
        return data.zip(label)

    def _transform_col_mp(self, d, col, attr, workers=0):
        if workers:
            out = np.array_split(np.array(d[col]), workers)
            result_set = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures_to_index = {
                    executor.submit(self._transform_col, j, col, attr): i
                    for i, j in enumerate(out)
                }
                for future in as_completed(futures_to_index):
                    index = futures_to_index[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(e)
                    else:
                        result_set.append((index, result))
                        result_set.sort(key=lambda x: x[0])
                        return np.vstack([i[1] for i in result_set])
        else:
            out = np.array(d[col])
            return self._transform_col(out, col, attr)

    def _transform_col(self, out, col, attr):
        if attr['type'] == 'dense':
            out[np.where(out == '')] = 0
            if col in self.encoders:
                out = self.encoders[col].transform(
                    out.astype(attr['dtype']).reshape(-1, 1)
                )
        elif attr['type'] == 'categorical':
            out[np.where(out == '')] = '-1'
            if col in self.encoders:
                out = self.encoders[col].encode_category(out.astype(attr['dtype']))
        else:
            # out = out.reshape(-1, 1)
            pass
        return out

    def train_dataloader(self):
        return DataLoader(
            self.data_train.sharding_filter(), batch_size=self.batch_size, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.var_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.var_batch_size)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self.data_cfg:
                return self.data_cfg[name]
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self).__name__, name)
            )

    def dump(self, path):
        joblib.dump(
            self.encoders, open(Path(path).joinpath('encoders.gz'), 'wb'), compress=3
        )
        json.dump(
            self.featuremap._asdict(), open(Path(path).joinpath('featuremap.json'), 'w')
        )

    def load(self, path):
        self.featuremap = FeatureMap(
            **json.load(open(Path(path).joinpath('featuremap.json'), 'r'))
        )
        self.encoders = joblib.load(open(Path(path).joinpath('encoders.gz'), 'rb'))


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
