import torch
import torch.nn as nn
from .basemodel import BaseModel

from ..layers import CIN


class xDeepFM(BaseModel):
    def __init__(
        self,
        featuremap,
        cin_layer_size=(256, 128),
        dnn_hidden_size=(256, 256),
        embedding_dim=4,
        dnn_dropout=0,
    ) -> None:
        super().__init__(featuremap, embedding_dim, linear=True)
        self.cin_layer_size = cin_layer_size
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_dropout = dnn_dropout
        self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]

        self.cin = CIN(self.sparse_features, cin_layer_size)
        self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False)

        dnn_input = int(self.dense_features + embedding_dim * self.sparse_features)
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
        dense_value_list = (
            [
                x[:, v['index'][0]].view(-1, 1)
                for v in self.features_attr.values()
                if v['type'] == 'dense'
            ]
            if self.featuremap.dense_features
            else 0
        )
        sparse_linear_embedding_list = [
            self.linear_embedding_dict[feat](x[:, v['index'][0]].view(-1, 1).long())
            for feat, v in self.features_attr.items()
            if v['type'] == 'categorical'
        ]

        sparse_embedding_cat = torch.cat(sparse_linear_embedding_list, dim=-1)
        sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
        dense_value_logit = (
            self.linear_dense(torch.cat(dense_value_list, dim=-1))
            if self.featuremap.dense_features
            else 0
        )
        linear_logit = sparse_feat_logit + dense_value_logit

        sparse_embedding_list = [
            self.embedding_dict[feat](x[:, v['index'][0]].view(-1, 1).long())
            for feat, v in self.features_attr.items()
            if v['type'] == 'categorical'
        ]
        cin_input = torch.cat(sparse_embedding_list, dim=1)
        cin_output = self.cin(cin_input)
        cin_logit = self.cin_linear(cin_output)

        if self.featuremap.dense_features:
            dnn_input = torch.cat(
                [
                    torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1),
                    torch.cat(dense_value_list, dim=1),
                ],
                dim=1,
            )
        else:
            dnn_input = torch.cat(
                [torch.flatten(torch.cat(sparse_embedding_list, dim=1), start_dim=1)],
                dim=1,
            )
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        final_logit = linear_logit + dnn_logit + cin_logit

        return final_logit
