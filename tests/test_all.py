'''
@File         :xdeepfm_test.py
@Description  :
@Time         :2022/12/22 17:58:06
@Author       :tangs
@Version      :1.0
'''

import sys

import pytest

sys.path.append('.')

import pytorch_lightning as pl
from src.dataset import BaseRecData, TorchRecData
from src.models import xDeepFM, IPNN


@pytest.mark.parametrize('model', ['xDeepFM', 'IPNN'])
def test_demo(model):
    data = TorchRecData('cfg/data/taobao.yml', 'features/featuremap')
    model = eval(model)(data.featuremap).double()

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
    )
    trainer.fit(model=model, datamodule=data)
    trainer.test(model, datamodule=data)
