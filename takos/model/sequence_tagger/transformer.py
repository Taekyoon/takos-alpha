import torch
import torch.nn as nn
from takos.model.modules.transform import Transformer
from takos.model.modules.rnn import CRF
from takos.configs.constants import START_TAG, STOP_TAG

from typing import Tuple, Dict


class TransformerTagger(nn.Module):
    def __init__(self, vocab_size: int, tag_size: int, embedding_dim: int, hidden_dim: int, head_size: int,
                 layer_size: int, pad_idx=0) -> None:
        super(TransformerTagger, self).__init__()
        self._pad_idx = pad_idx
        self._tag_size = tag_size

        self.transformer = Transformer(embedding_dim, hidden_dim * head_size, vocab_size, num_heads=head_size,
                                       num_layers=layer_size, causal=True)
        self._fc1 = nn.Linear(embedding_dim, embedding_dim)
        self._activate = nn.Tanh()
        self._fc2 = nn.Linear(embedding_dim, tag_size)

        self._ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx)

        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        linear_hidden = self._activate(self._fc1(hiddens))
        emissions = self._fc2(linear_hidden)

        emissions = nn.functional.softmax(emissions, dim=-1)
        path = torch.argmax(emissions, dim=-1)
        max_prob = torch.max(emissions, dim=-1)[0]
        scores = torch.mean(max_prob)

        return scores, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx)
        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        linear_hidden = self._activate(self._fc1(hiddens))
        emissions = self._fc2(linear_hidden)

        nll = self._ce_loss(emissions.view(-1, self._tag_size), y.view(-1))
        # nll = torch.mean(nll * masking.view(-1).float())
        nll = torch.mean(nll)

        return nll


class TransformerCRF(nn.Module):
    def __init__(self, vocab_size: int, tag_to_idx: Dict, embedding_dim: int, hidden_dim: int, head_size: int,
                 layer_size: int, pad_idx=0) -> None:
        super(TransformerCRF, self).__init__()
        self._pad_idx = pad_idx
        self._tag_to_idx = tag_to_idx

        self.transformer = Transformer(embedding_dim, hidden_dim, vocab_size, num_heads=head_size,
                                       num_layers=layer_size)
        self._fc = nn.Linear(embedding_dim, len(tag_to_idx))

        self._crf = CRF(len(self._tag_to_idx), bos_tag_id=self._tag_to_idx[START_TAG],
                        eos_tag_id=self._tag_to_idx[STOP_TAG],
                        pad_tag_id=self._pad_idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx)
        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        emissions = self._fc(hiddens)

        score, path = self._crf.decode(emissions, mask=masking.float())

        return score, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx)

        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        emissions = self._fc(hiddens)

        nll = self._crf(emissions, y, mask=masking.float())

        return nll
