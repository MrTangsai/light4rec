'''
@File         :datapipe.py
@Description  :
@Time         :2022/12/15 11:14:02
@Author       :tangs
@Version      :1.0
'''

import torch.utils.data as Data
import torchdata.datapipes as dp


class RecIterDataPipe(Data.IterDataPipe):
    def __init__(self, paths, sep, row_mapper) -> None:
        super().__init__()
        self.paths = paths
        self.sep = sep
        self.row_mapper = row_mapper

    def __iter__(self):
        # worker_info = Data.get_worker_info()
        # paths = self.paths
        # if worker_info is not None:
        #     paths = (
        #         path
        #         for (idx, path) in enumerate(paths)
        #         if idx % worker_info.num_workers == worker_info.id
        #     )
        datapipe = (
            dp.iter.FileOpener(self.paths, mode='rt')
            .parse_csv(delimiter=self.sep, skip_lines=1)
            .map(self.row_mapper)
        )
        yield from datapipe


class RecIterDataPipe(Data.IterDataPipe):
    def __init__(self, paths, sep, row_mapper) -> None:
        super().__init__()
        self.paths = paths
        self.sep = sep
        self.row_mapper = row_mapper

    def __iter__(self):
        datapipe = dp.iter.FileOpener(self.paths, mode='rt').readlines(
            return_path=False
        )
        yield from datapipe
