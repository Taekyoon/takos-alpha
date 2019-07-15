import torch
from takos.model.sequence_tagger.transformer import TransformerTagger, TransformerCRF


from takos.configs.constants import PAD, START_TAG, STOP_TAG


def test_train_transformer_tagger():
    vocab_size = 100
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32
    head_size = 8
    layer_size = 4

    transformer = TransformerTagger(vocab_size, len(tag_to_idx), embedding_size, hidden_size, head_size, layer_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()
    targets = torch.randint(3, 5, (12, 20)).long()

    loss = transformer.loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)


def test_infer_transformer_tagger():
    vocab_size = 100
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32
    head_size = 8
    layer_size = 4

    transformer = TransformerTagger(vocab_size, len(tag_to_idx), embedding_size, hidden_size, head_size, layer_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()

    outputs = transformer(inputs)

    assert outputs[1].size() == (12, 20)


def test_train_transformer_crf():
    vocab_size = 100
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32
    head_size = 8
    layer_size = 4

    transformer = TransformerCRF(vocab_size, tag_to_idx, embedding_size, hidden_size, head_size, layer_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()
    targets = torch.randint(3, 5, (12, 20)).long()

    loss = transformer.loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)


def test_infer_transformer_crf():
    vocab_size = 100
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32
    head_size = 8
    layer_size = 4

    transformer = TransformerCRF(vocab_size, tag_to_idx, embedding_size, hidden_size, head_size, layer_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()

    outputs = transformer(inputs)

    assert outputs[1].size() == (12, 20)
