from typing import Tuple, Dict
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


class JointClsNTagDatasetFromJSONFile(Dataset):
    def __init__(self,
                 json_path: str,
                 enable_length: bool = True,
                 limit_pad_len: int = None,
                 pad_value: int = 0) -> None:
        dataset = json.load(open(json_path, 'rb'))

        self._inputs = np.array(dataset['inputs'])
        self._slots = np.array(dataset['slots'])
        self._intents = np.array(dataset['intents'])

        self.enable_length = enable_length
        self.limit_pad_len = limit_pad_len
        self.pad_value = pad_value

        return

    def __len__(self) -> int:
        len_dataset = len(self._inputs)

        return len_dataset

    def __getitem__(self, idx: int) -> Dict:
        sampled_instances = dict()
        sampled_instances['inputs'] = dict()

        sampled_inputs = self._inputs[idx]
        sampled_slots = self._slots[idx]
        sampled_intents = self._intents[idx]

        if self.enable_length:
            if isinstance(sampled_inputs[0], list):
                inputs_length = [len(inst) for inst in sampled_inputs]
            else:
                inputs_length = [len(sampled_inputs)]

            if self.limit_pad_len is not None:
                inputs_length = [l if l < self.limit_pad_len else self.limit_pad_len for l in inputs_length]
            sampled_instances['inputs']['length'] = Tensor(inputs_length).long()

        if self.limit_pad_len is not None:
            sampled_inputs = pad_sequences(sampled_inputs, self.limit_pad_len, pad_value=self.pad_value)
            sampled_slots = pad_sequences(sampled_slots, self.limit_pad_len, pad_value=self.pad_value)

        sampled_instances['inputs']['value'] = Tensor(sampled_inputs).long()
        sampled_instances['slots'] = Tensor(sampled_slots).long()
        sampled_instances['intents'] = Tensor(sampled_intents).long()

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


class SequencePairDatasetFromJSONFile(Dataset):
    def __init__(self,
                 json_path: str,
                 enable_length: bool = True,
                 limit_src_pad_len: int = None,
                 limit_tgt_pad_len: int = None,
                 pad_value: int = 0) -> None:
        dataset = json.load(open(json_path, 'rb'))

        self._sources = dataset['sources']
        self._targets = dataset['targets']

        self.enable_length = enable_length
        self.limit_src_pad_len = limit_src_pad_len
        self.limit_tgt_pad_len = limit_tgt_pad_len
        self.pad_value = pad_value

        return

    def __len__(self) -> int:
        len_dataset = len(self._sources)

        return len_dataset

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sampled_instances = dict()
        sampled_instances['sources'] = dict()

        sampled_sources = self._sources[idx]
        sampled_targets = self._targets[idx]

        if self.enable_length:
            if isinstance(sampled_sources[0], list):
                sources_length = [len(inst) for inst in sampled_sources]
            else:
                sources_length = [len(sampled_sources)]

            if isinstance(sampled_targets[0], list):
                targets_length = [len(inst) for inst in sampled_targets]
            else:
                targets_length = [len(sampled_targets)]

            if self.limit_src_pad_len is not None:
                sources_length = [l if l < self.limit_src_pad_len else self.limit_src_pad_len for l in sources_length]
            sampled_instances['sources']['length'] = Tensor(sources_length).long()

            if self.limit_tgt_pad_len is not None:
                targets_length = [l if l < self.limit_tgt_pad_len else self.limit_tgt_pad_len for l in targets_length]
            sampled_instances['targets']['length'] = Tensor(targets_length).long()

        if self.limit_src_pad_len is not None:
            sampled_sources = pad_sequences(sampled_sources, self.limit_src_pad_len, self.pad_value)
            sampled_targets = pad_sequences(sampled_targets, self.limit_tgt_pad_len, self.pad_value)

        sampled_instances['sources']['value'] = Tensor(sampled_sources).long()
        sampled_instances['targets']['value'] = Tensor(sampled_targets).long()

        return sampled_instances
