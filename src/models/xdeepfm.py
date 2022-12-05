import torch
import torch.nn as nn
from .basemodel import BaseModel

from ..layers import CIN


class xDeepFM(BaseModel):
    def __init__(
        self,
        data_cfg,
        cin_layer_size=(256, 128),
        dnn_hidden_size=(256, 256),
        embedding_dim=4,
        dnn_dropout=0,
    ) -> None:
        super().__init__(data_cfg, embedding_dim, linear=True)
        self.cin_layer_size = cin_layer_size
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_dropout = dnn_dropout
        self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]

        self.cin = CIN(len(self.sparse_features), cin_layer_size)
        self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False)

        dnn_input = int(
            len(self.dense_features) + embedding_dim * len(self.sparse_features)
        )
        self.dnn_hidden_size = (dnn_input,) + self.dnn_hidden_size
        dnn_hidden_size_tuple = list(
            zip(self.dnn_hidden_size[:-1], self.dnn_hidden_size[1:])
        )
        self.dnn = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(*size), nn.ReLU(), nn.Dropout(self.dnn_dropout))
                for size in dnn_hidden_size_tuple
            ]
        )
        self.dnn_linear = nn.Linear(self.dnn_hidden_size[-1], 1, bias=False)

    def forward(self, x):
        dense_value_list = [
            x[:, index].view(-1, 1) for index in self.dense_features.values()
        ]
        sparse_linear_embedding_list = [
            self.linear_embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.sparse_features.items()
        ]

        sparse_embedding_cat = torch.cat(sparse_linear_embedding_list, dim=-1)
        sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
        dense_value_logit = self.linear_dense(torch.cat(dense_value_list, dim=-1))
        linear_logit = sparse_feat_logit + dense_value_logit

        sparse_embedding_list = [
            self.embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.sparse_features.items()
        ]
        cin_input = torch.cat(sparse_embedding_list, dim=1)
        cin_output = self.cin(cin_input)
        cin_logit = self.cin_linear(cin_output)

        dnn_input = torch.cat(
            [
                torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1),
                torch.cat(dense_value_list, dim=1),
            ],
            dim=1,
        )
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        final_logit = linear_logit + dnn_logit + cin_logit

        return final_logit
