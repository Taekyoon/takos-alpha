import torch
import torch.nn as nn
from torch.nn import functional as F

from takos.model.modules.rnn import BiLSTM, CRF
from takos.configs.constants import START_TAG, STOP_TAG

from typing import Tuple, Dict


class BilstmCRF(nn.Module):
    """BilstmCRF"""
    def __init__(self, vocab_size: int, tag_to_idx: Dict, embedding_dim: int, hidden_dim: int, pad_idx=0) -> None:
        super(BilstmCRF, self).__init__()
        self._pad_idx = pad_idx
        self._tag_to_idx = tag_to_idx

        self._embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._bilstm = BiLSTM(embedding_dim, hidden_dim)
        self._fc = nn.Linear(2 * hidden_dim, len(tag_to_idx))

        self._crf = CRF(len(self._tag_to_idx), bos_tag_id=self._tag_to_idx[START_TAG],
                        eos_tag_id=self._tag_to_idx[STOP_TAG],
                        pad_tag_id=self._pad_idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)
        hiddens, _ = self._bilstm(fmap, masking)
        emissions = self._fc(hiddens)
        score, path = self._crf.decode(emissions, mask=masking)
        return score, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)
        hiddens, _ = self._bilstm(fmap, masking)
        emissions = self._fc(hiddens)
        nll = self._crf(emissions, y, mask=masking)
        return nll

    def get_probs(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)
        hiddens, _ = self._bilstm(fmap, masking)
        emissions = self._fc(hiddens)
        probs = F.softmax(emissions[:, :, 3:], dim=-1)

        return probs.cpu().detach().numpy()
