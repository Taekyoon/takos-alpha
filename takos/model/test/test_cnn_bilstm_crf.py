import torch
from takos.model.sequence_tagger.cnn_bilstm_crf import CNNBilstmCRF

from takos.configs.constants import PAD, START_TAG, STOP_TAG


def test_train_cnn_bi_lstm_crf():
    vocab_size = 100
    channel_dim = 32
    conv_configs = [{'channel_size': 16, 'kernel_size': 3, 'padding': 1},
                    {'channel_size': 16, 'kernel_size': 5, 'padding': 2}]
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32

    cnn_bilstm_crf = CNNBilstmCRF(vocab_size, tag_to_idx, embedding_size, channel_dim, conv_configs, hidden_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()
    targets = torch.randint(3, 5, (12, 20)).long()

    loss = cnn_bilstm_crf.loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)


def test_infer_cnn_bi_lstm_crf():
    vocab_size = 100
    channel_dim = 32
    conv_configs = [{'channel_size': 16, 'kernel_size': 3, 'padding': 1},
                    {'channel_size': 16, 'kernel_size': 5, 'padding': 2}]
    embedding_size = 32
    tag_to_idx = {PAD: 0, START_TAG: 1, STOP_TAG: 2, 'B': 3, 'I': 4, 'O': 5}
    hidden_size = 32

    cnn_bilstm_crf = CNNBilstmCRF(vocab_size, tag_to_idx, embedding_size, channel_dim, conv_configs, hidden_size)

    inputs = torch.randint(0, vocab_size, (12, 20)).long()

    outputs = cnn_bilstm_crf(inputs)

    assert outputs[1].size() == (12, 20)
