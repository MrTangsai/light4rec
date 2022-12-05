'''
@File         :din.py
@Description  :
@Time         :2022/12/01 11:14:34
@Author       :tangs
@Version      :1.0
'''

import torch
import torch.nn as nn
from .basemodel import SquenceModel


class DIN(SquenceModel):
    def __init__(self, data_dir, label_name='label', embedding_dim=4) -> None:
        super(DIN, self).__init__(data_dir, label_name, embedding_dim)
        dnn_input = int(
            len(self.dense_features)
            + len(self.sparse_features) * (len(self.sparse_features) - 1) / 2
            + embedding_dim * len(self.sparse_features)
        )
        self.dnn = nn.Sequential(nn.Linear(dnn_input, 128), nn.Linear(128, 128))
        self.dnn_linear = nn.Linear(128, 1)

    def forward(self, x):
        dense_value_list = [
            x[:, index].view(-1, 1) for index in self.dense_features.values()
        ]

        query_emb_list = [
            self.embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.din_target_field.items()
        ]
        keys_emb_list = [
            self.embedding_dict[feat](x[:, index[1:]].long())
            for feat, index in self.din_sequence_field.items()
        ]
        dnn_input_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            to_list=True,
        )
