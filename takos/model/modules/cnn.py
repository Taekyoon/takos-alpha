import torch
from torch import nn
from torch.nn.modules import Conv1d, Linear, Dropout, BatchNorm1d
from torch.nn import functional as F


class MultiCNN(nn.Module):
    def __init__(self, input_size, output_size, convs_configs, dropout=.5, normalize=True):
        super(MultiCNN, self).__init__()
        input_size = input_size
        concat_cnn_output_size = 0
        output_size = output_size

        self._normalize = normalize

        self.cnn_modules = nn.ModuleList()

        for configs in convs_configs:
            channel_size = configs['channel_size']
            kernel_size = configs['kernel_size']
            padding = configs['padding']

            concat_cnn_output_size += channel_size

            module = Conv1d(input_size, channel_size, kernel_size=kernel_size, padding=padding)
            self.cnn_modules.append(module)

        self.batch_normlize = BatchNorm1d(concat_cnn_output_size)

        self.dropout = Dropout(dropout)
        self.output_linear = Linear(concat_cnn_output_size, output_size)

    def forward(self, inputs):
        cnn_outputs = list()

        inputs = inputs.transpose(-2, -1)

        for cnn_module in self.cnn_modules:
            cnn_outputs.append(F.relu(cnn_module(inputs)))

        concat_cnn_outputs = torch.cat(cnn_outputs, dim=-2)

        if self._normalize:
            concat_cnn_outputs = self.batch_normlize(concat_cnn_outputs)

        concat_cnn_outputs = self.dropout(concat_cnn_outputs)
        concat_cnn_outputs = concat_cnn_outputs.transpose(-2, -1)

        outputs = self.output_linear(concat_cnn_outputs)
        return outputs
