'''
@File         :basedata.py
@Description  :data base class 
@Time         :2022/11/14 23:30:21
@Author       :tangs
@Version      :1.0
'''
import pandas as pd
import pytorch_lightning as pl


class BaseRecData(pl.LightningDataModule):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir

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
