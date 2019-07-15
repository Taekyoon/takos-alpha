from typing import Tuple
import json

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset


class SequenceTagDatasetFromJSONFile(Dataset):
    def __init__(self,
                 json_path: str,
                 enable_length: bool = True,
                 limit_pad_len: int = None,
                 pad_value: int = 0) -> None:
        dataset = json.load(open(json_path, 'rb'))

        self._inputs = dataset['inputs']
        self._entities = dataset['entities']

        self.enable_length = enable_length
        self.limit_pad_len = limit_pad_len
        self.pad_value = pad_value

        return

    def __len__(self) -> int:
        len_dataset = len(self._inputs)

        return len_dataset

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sampled_instances = dict()
        sampled_instances['inputs'] = dict()

        sampled_inputs = self._inputs[idx]
        sampled_entities = self._entities[idx]

        if self.enable_length:
            if isinstance(sampled_inputs[0], list):
                inputs_length = [len(inst) for inst in sampled_inputs]
            else:
                inputs_length = [len(sampled_inputs)]

            if self.limit_pad_len is not None:
                inputs_length = [l if l < self.limit_pad_len else self.limit_pad_len for l in inputs_length]
            sampled_instances['inputs']['length'] = Tensor(inputs_length).long()

        if self.limit_pad_len is not None:
            sampled_inputs = pad_sequences(sampled_inputs, self.limit_pad_len, self.pad_value)
            sampled_entities = pad_sequences(sampled_entities, self.limit_pad_len, self.pad_value)

        sampled_instances['inputs']['value'] = Tensor(sampled_inputs).long()
        sampled_instances['entities'] = Tensor(sampled_entities).long()

        return sampled_instances


def pad_sequences(dataset, limit_len, pad_value=0):
    if isinstance(dataset[0], list):
        batch_size = len(dataset)
    else:
        batch_size = 1
        dataset = [dataset]

    padded_sequences = np.full((batch_size, limit_len), pad_value)

    for i, inst in enumerate(dataset):
        len_inst = len(inst)

        if len_inst > limit_len:
            padded_sequences[i, :] = np.array(inst[:limit_len])
        elif len_inst <= limit_len:
            padded_sequences[i, :len_inst] = np.array(inst[:len_inst])

    return padded_sequences
