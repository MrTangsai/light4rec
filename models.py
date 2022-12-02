'''
@File         :models.py
@Description  :
@Time         :2022/11/12 18:47:51
@Author       :tangs
@Version      :1.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, df, embedding_dim, linear=False) -> None:
        super(BaseModel, self).__init__()
        self.columns = df.columns
        self.sparse_features = {
            j: i for i, j in enumerate(self.columns) if j.startswith('C')
        }
        self.dense_features = {
            j: i for i, j in enumerate(self.columns) if j.startswith('I')
        }

        self.embedding_dict = nn.ModuleDict(
            {
                feat: nn.Embedding(
                    num_embeddings=int(df[feat].max() + 1),
                    embedding_dim=embedding_dim,
                )
                for feat in self.sparse_features.keys()
            }
        )
        self.linear = linear
        if self.linear:
            self.linear_embedding_dict = nn.ModuleDict(
                {
                    feat: nn.Embedding(
                        num_embeddings=int(df[feat].max() + 1),
                        embedding_dim=1,
                    )
                    for feat in self.sparse_features.keys()
                }
            )
            self.linear_dense = nn.Linear(len(self.dense_features), 1, bias=False)

    def forward(self):
        NotImplemented


class IPNN(BaseModel):
    def __init__(self, df, embedding_dim=4) -> None:
        super(IPNN, self).__init__(df, embedding_dim)
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

        sparse_embedding_list = [
            self.embedding_dict[feat](x[:, index].view(-1, 1).long())
            for feat, index in self.sparse_features.items()
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

        dnn_input = torch.cat(
            [product_layer, torch.cat(dense_value_list, dim=1)], dim=1
        )

        dnn_output = self.dnn(dnn_input)

        dnn_logit = self.dnn_linear(dnn_output)

        # nn.BCEWithLogitsLoss = nn.BCELoss+nn.Sigmoid
        # y_pred = torch.sigmoid(dnn_logit)

        return dnn_logit


class CIN(nn.Module):
    def __init__(self, init_size, cin_layer_size=(256, 128)) -> None:
        super(CIN, self).__init__()
        self.cin_layer_size = cin_layer_size
        self.cin_convs = nn.ModuleList()
        self.cin_convs.append(
            nn.Conv1d(init_size**2, self.cin_layer_size[0], kernel_size=1)
        )
        for index, size in enumerate(self.cin_layer_size[1:]):
            self.cin_convs.append(
                nn.Conv1d(
                    int(init_size * self.cin_layer_size[index] / 2),
                    int(size),
                    kernel_size=1,
                )
            )

    def forward(self, x):
        cross_layer, hidden_layer = [x], [x]
        batch_size, _, dim = x.size()

        for i, size in enumerate(self.cin_layer_size):
            hidden_output = self.cin_convs[i](
                torch.einsum('abc,adc->abdc', cross_layer[0], cross_layer[-1]).view(
                    batch_size, -1, dim
                )
            )
            hidden_output = F.relu(hidden_output)
            if i == len(self.cin_layer_size) - 1:
                hidden_layer.append(hidden_output)
            else:
                h1, h2 = torch.split(hidden_output, size // 2, 1)
                cross_layer.append(h1)
                hidden_layer.append(h2)

        output = torch.sum(torch.cat(hidden_layer[1:], 1), -1)
        return output


class xDeepFM(BaseModel):
    def __init__(
        self,
        df,
        cin_layer_size=(256, 128),
        dnn_hidden_size=(256, 256),
        embedding_dim=4,
        dnn_dropout=0,
    ) -> None:
        """xDeepFM class for recommendation

        Args:
            df (pd.DataFrame): pandas DataFrame
            cin_layer_size (tuple, optional): the size of each cin layer
            dnn_hidden_size (tuple, optional): the size of each hidden layer
            embedding_dim (int, optional): the size of each embedding vector
            dnn_dropout (float, optional): float in [0,1), the probability we will drop out a given DNN coordinate.

        """

        super(xDeepFM, self).__init__(df, embedding_dim, linear=True)
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
        # y_pred = torch.sigmoid(final_logit)
        return final_logit
