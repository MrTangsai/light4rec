'''
@File         :datapipe.py
@Description  :
@Time         :2022/12/15 11:14:02
@Author       :tangs
@Version      :1.0
'''

import random
import numpy as np
import torch.utils.data as Data
import torchdata.datapipes as dp
from torchdata.datapipes import functional_datapipe
from functools import partial
from typing import Any, Callable, Iterator, Tuple


class CSVReader(Data.IterDataPipe):
    def __init__(self, paths, sep, row_mapper=None) -> None:
        super().__init__()
        paths = paths if isinstance(paths, list) else [paths]
        paths = [str(i) for i in paths]
        self.paths = paths
        self.sep = sep

    def __iter__(self):
        # worker_info = Data.get_worker_info()
        # paths = self.paths
        # if worker_info is not None:
        #     paths = (
        #         path
        #         for (idx, path) in enumerate(paths)
        #         if idx % worker_info.num_workers == worker_info.id
        #     )
        datapipe = dp.iter.FileOpener(self.paths, mode='rt').parse_csv(
            delimiter=self.sep, skip_lines=1
        )
        # if self.row_mapper:
        #     datapipe = datapipe.map(self.row_mapper)
        yield from datapipe

    def __len__(self):
        count = 0
        for _ in iter(self):
            count += 1
        return count


class CSVHeader(Data.IterDataPipe):
    def __init__(self, paths, sep) -> None:
        super().__init__()
        paths = paths if isinstance(paths, list) else [paths]
        paths = [str(i) for i in paths]
        self.paths = paths
        self.sep = sep

    def __iter__(self):
        datapipe = (
            dp.iter.FileOpener(self.paths, mode='rt')
            .readlines(return_path=False)
            .map(lambda x: x.split(self.sep))
        )
        yield from datapipe


class NPYReader(Data.IterDataPipe):
    def __init__(self, paths) -> None:
        super().__init__()
        self.paths = str(paths)
        self.datapipe = dp.iter.FileLister(self.paths).filter(
            lambda x: x.endswith('npy')
        )

    def __iter__(self):
        for file in self.datapipe:
            yield from np.load(file)


class _RandFilter(Data.IterDataPipe):
    def __init__(
        self,
        datapipe: Data.IterDataPipe,
        filter_fn: Callable[[random.Random], bool],
        rand_gen: random.Random,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.filter_fn = filter_fn
        self.rand_gen = rand_gen
        self.rand_gen_init_state: Tuple[Any, ...] = rand_gen.getstate()

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        self.rand_gen.setstate(self.rand_gen_init_state)
        for data in self.datapipe:
            if self.filter_fn(self.rand_gen):
                yield data


def _rand_train_filter_fn(
    train_perc: float,
    rand_gen: random.Random,
) -> bool:
    return rand_gen.random() < train_perc


def _rand_val_filter_fn(train_perc: float, rand_gen: random.Random) -> bool:
    return not _rand_train_filter_fn(train_perc, rand_gen)


def rand_split_train_val(
    datapipe: Data.IterDataPipe,
    train_perc: float,
    random_seed: int = 0,
) -> Tuple[Data.IterDataPipe, Data.IterDataPipe]:
    if not 0.0 < train_perc < 1.0:
        raise ValueError("train_perc must be in range (0.0, 1.0)")

    return _RandFilter(
        datapipe, partial(_rand_train_filter_fn, train_perc), random.Random(random_seed)
    ), _RandFilter(
        datapipe, partial(_rand_val_filter_fn, train_perc), random.Random(random_seed)
    )
