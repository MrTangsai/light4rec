'''
@File         :ipnn.py
@Description  :
@Time         :2022/12/01 11:14:10
@Author       :tangs
@Version      :1.0
'''


import torch
import torch.nn as nn
from .basemodel import BaseModel


class IPNN(BaseModel):
    def __init__(self, featuremap, embedding_dim=4) -> None:
        super(IPNN, self).__init__(featuremap, embedding_dim)
        dnn_input = int(
            self.dense_features
            + self.sparse_features * (self.sparse_features - 1) / 2
            + embedding_dim * self.sparse_features
        )
        self.dnn = nn.Sequential(nn.Linear(dnn_input, 128), nn.Linear(128, 128))
        self.dnn_linear = nn.Linear(128, 1)

    def forward(self, x):
        dense_value_list = (
            [
                x[:, v['index'][0]].view(-1, 1)
                for v in self.features_attr.values()
                if v['type'] == 'dense'
            ]
            if self.featuremap.dense_features
            else 0
        )

        sparse_embedding_list = [
            self.embedding_dict[feat](x[:, v['index'][0]].view(-1, 1).long())
            for feat, v in self.features_attr.items()
            if v['type'] == 'categorical'
        ]

        linear_signal = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1
        )

        num_inputs = len(sparse_embedding_list)

        row, col = [], []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat(
            [sparse_embedding_list[idx] for idx in row], dim=1
        )  # batch num_pairs k
        q = torch.cat([sparse_embedding_list[idx] for idx in col], dim=1)

        inner_product = torch.flatten(
            torch.sum(p * q, dim=2, keepdim=True), start_dim=1
        )

        product_layer = torch.cat([linear_signal, inner_product], dim=1)

        dnn_input = (
            torch.cat([product_layer, torch.cat(dense_value_list, dim=1)], dim=1)
            if self.featuremap.dense_features
            else torch.cat([product_layer], dim=1)
        )

        dnn_output = self.dnn(dnn_input)

        dnn_logit = self.dnn_linear(dnn_output)

        # nn.BCEWithLogitsLoss = nn.BCELoss+nn.Sigmoid
        # y_pred = torch.sigmoid(dnn_logit)

        return dnn_logit
