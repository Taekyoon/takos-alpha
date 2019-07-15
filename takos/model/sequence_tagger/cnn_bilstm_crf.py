import torch
import torch.nn as nn
from takos.model.modules.rnn import BiLSTM, CRF
from takos.model.modules.cnn import MultiCNN
from takos.configs.constants import START_TAG, STOP_TAG

from typing import Tuple, Dict


class CNNBilstmCRF(nn.Module):
    """BilstmCRF"""
    def __init__(self, vocab_size: int, tag_to_idx: Dict, embedding_dim: int, channel_dim: int, convs_configs: Dict,
                 hidden_dim: int, pad_idx=0) -> None:
        super(CNNBilstmCRF, self).__init__()
        self._pad_idx = pad_idx
        self._tag_to_idx = tag_to_idx

        self._embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._multi_conv = MultiCNN(embedding_dim, channel_dim, convs_configs)
        self._bilstm = BiLSTM(channel_dim, hidden_dim)
        self._fc = nn.Linear(2 * hidden_dim, len(tag_to_idx))

        self._crf = CRF(len(self._tag_to_idx), bos_tag_id=self._tag_to_idx[START_TAG],
                        eos_tag_id=self._tag_to_idx[STOP_TAG],
                        pad_tag_id=self._pad_idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)
        conv_outputs = self._multi_conv(fmap)
        hiddens, _ = self._bilstm(conv_outputs, masking)
        emissions = self._fc(hiddens)
        score, path = self._crf.decode(emissions, mask=masking)
        return score, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)
        conv_outputs = self._multi_conv(fmap)
        hiddens, _ = self._bilstm(conv_outputs, masking)
        emissions = self._fc(hiddens)
        nll = self._crf(emissions, y, mask=masking)
        return nll
