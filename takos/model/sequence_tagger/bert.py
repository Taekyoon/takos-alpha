import torch

import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from typing import Tuple


class BertTagger(BertPreTrainedModel):
    def __init__(self, config, tag_size, pad_idx=0, cls_idx=2) -> None:
        super(BertTagger, self).__init__(config)
        self._pad_idx = pad_idx
        self._cls_idx = cls_idx
        self._tag_size = tag_size

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_2 = nn.Linear(config.hidden_size, tag_size)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.apply(self.init_bert_weights)

        self.device = torch.device('cpu')

    def to(self, device):
        super(BertTagger, self).to(device)
        self.device = device

        return

    def forward(self, x: torch.Tensor)-> Tuple[torch.tensor, torch.tensor]:
        batch_size = x.size(0)
        cls = torch.full((batch_size, 1), self._cls_idx).to(self.device).long()
        x = torch.cat([x, cls], dim=-1)

        masking = x.ne(self._pad_idx)

        encoder_layer, _ = self.bert(x, attention_mask=masking, output_all_encoded_layers=False)
        encoder_layer = encoder_layer[:, 1:]

        dense_layer = self.dense(encoder_layer)
        dense_layer = self.activation(dense_layer)
        emissions = self.dense_2(dense_layer)

        emissions = nn.functional.softmax(emissions, dim=-1)
        path = torch.argmax(emissions, dim=-1)
        max_prob = torch.max(emissions, dim=-1)[0]
        scores = torch.mean(max_prob)

        return scores, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls = torch.full((batch_size, 1), self._cls_idx).to(self.device).long()
        x = torch.cat([x, cls], dim=-1)

        masking = x.ne(self._pad_idx)

        encoder_layer, _ = self.bert(x, attention_mask=masking, output_all_encoded_layers=False)
        encoder_layer = encoder_layer[:, 1:]

        dense_layer = self.dense(encoder_layer)
        dense_layer = self.activation(dense_layer)
        emissions = self.dense_2(dense_layer)

        nll = self.ce_loss(emissions.view(-1, self._tag_size), y.view(-1))
        # nll = torch.mean(nll * masking[:, 1:].contiguous().view(-1).float())
        nll = torch.mean(nll)

        return nll

    def get_probs(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        batch_size = x.size(0)
        cls = torch.full((batch_size, 1), self._cls_idx).to(self.device).long()
        x = torch.cat([x, cls], dim=-1)

        masking = x.ne(self._pad_idx)

        encoder_layer, _ = self.bert(x, attention_mask=masking, output_all_encoded_layers=False)
        encoder_layer = encoder_layer[:, 1:]

        dense_layer = self.dense(encoder_layer)
        dense_layer = self.activation(dense_layer)
        emissions = self.dense_2(dense_layer)
        probs = F.softmax(emissions[:, :, 3:], dim=-1)

        return probs.cpu().detach().numpy()
