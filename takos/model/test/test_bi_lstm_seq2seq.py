import torch
from takos.model.seq2seq import BiLSTMSeq2Seq, BiLSTMEncoder, LSTMDecoder


def test_train_bi_lstm_seq2seq():
    src_vocab_size = 100
    tgt_vocab_size = 100
    embedding_size = 32
    hidden_size = 32

    encoder = BiLSTMEncoder(src_vocab_size, embedding_size, hidden_size)
    decoder = LSTMDecoder(tgt_vocab_size, embedding_size, hidden_size * 2, hidden_size)
    seq2seq = BiLSTMSeq2Seq(encoder, decoder)

    inputs = torch.randint(0, src_vocab_size, (12, 20)).long()
    targets = torch.randint(0, tgt_vocab_size, (12, 20)).long()

    loss = seq2seq.loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)


def test_infer_bi_lstm_seq2seq():
    src_vocab_size = 100
    tgt_vocab_size = 100
    embedding_size = 32
    hidden_size = 32

    encoder = BiLSTMEncoder(src_vocab_size, embedding_size, hidden_size)
    decoder = LSTMDecoder(tgt_vocab_size, embedding_size, hidden_size * 2, hidden_size)
    seq2seq = BiLSTMSeq2Seq(encoder, decoder)

    inputs = torch.randint(0, src_vocab_size, (1, 20)).long()

    outputs = seq2seq(inputs)

    assert inputs.size() == outputs.size()
