import torch
import pytorch_lightning as pl
from src.dataset import BaseRecData, TorchRecData
from src.models import xDeepFM, IPNN

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

data = TorchRecData('cfg/data/taobao.yml', 'features/featuremap')
model = IPNN(data.featuremap).double()

# train model
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=1,
    limit_train_batches=10,
    limit_val_batches=10,
    limit_test_batches=10,
    callbacks=[
        progress_bar,
        RichModelSummary(),
        EarlyStopping(monitor="val_loss", mode="min"),
    ],
)
trainer.fit(model=model, datamodule=data)
trainer.test(model, datamodule=data)
