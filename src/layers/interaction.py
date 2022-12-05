'''
@File         :interaction.py
@Description  :
@Time         :2022/11/29 10:07:27
@Author       :tangs
@Version      :1.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


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
