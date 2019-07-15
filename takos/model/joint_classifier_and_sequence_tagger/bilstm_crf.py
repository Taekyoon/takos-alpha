import torch
import torch.nn as nn
from takos.model import BiLSTM, CRF
from takos.configs.constants import START_TAG, STOP_TAG

from typing import Tuple, Dict


class BilstmCRF(nn.Module):
    """BilstmCRF"""
    def __init__(self, vocab_size: int, class_size: int, tag_to_idx: Dict, embedding_dim: int, hidden_dim: int, pad_idx=0) -> None:
        """Instantiating BilstmCRF class
        Args:
            token_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has token information
            label_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has label information
            lstm_hidden_dim (int): the number of hidden dimension of lstm
        """
        super(BilstmCRF, self).__init__()
        self._pad_idx = pad_idx
        self._tag_to_idx = tag_to_idx

        self._embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._bilstm = BiLSTM(embedding_dim, hidden_dim)
        self._fc = nn.Linear(2 * hidden_dim, len(tag_to_idx))
        self._fc_2 = nn.Linear(2 * hidden_dim, class_size)

        self._crf = CRF(len(self._tag_to_idx), bos_tag_id=self._tag_to_idx[START_TAG],
                        eos_tag_id=self._tag_to_idx[STOP_TAG],
                        pad_tag_id=self._pad_idx)

        self._ce_loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)

        hiddens, last_hidden = self._bilstm(fmap, masking)
        last_hidden = torch.cat([*last_hidden[0]], dim=1)

        emissions = self._fc(hiddens)
        class_outputs = self._fc_2(last_hidden)
        class_probs = nn.functional.softmax(class_outputs, dim=-1)
        score, path = self._crf.decode(emissions, mask=masking)

        return score, path, class_probs

    def loss(self, x: torch.Tensor, y:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx).float()
        fmap = self._embedding(x)

        hiddens, last_hidden = self._bilstm(fmap, masking)
        last_hidden = torch.cat([*last_hidden[0]], dim=1)

        emissions = self._fc(hiddens)
        class_outputs = self._fc_2(last_hidden)
        class_loss = self._ce_loss(class_outputs, c.squeeze(-1))

        nll = self._crf(emissions, y, mask=masking)

        return nll, class_loss
