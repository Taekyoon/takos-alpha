import numpy as np

from takos.data_manager.dataset import pad_sequences


def test_pad_sequences_without_pad_val():
    limit_len = 3
    dummy_dataset = [[1, 2, 3, 1],
                     [1],
                     [1, 2, 3],
                     [1, 2],
                     [1, 2, 3]]
    np_dummy_dataset = np.array(dummy_dataset)

    answer_dataset = [[1, 2, 3],
                      [1, 0, 0],
                      [1, 2, 3],
                      [1, 2, 0],
                      [1, 2, 3]]
    np_answer_dataset = np.array(answer_dataset)

    padded_dataset = pad_sequences(np_dummy_dataset, limit_len)

    assert padded_dataset.all() == np_answer_dataset.all()


def test_pad_sequences_with_pad_val():
    limit_len = 3
    pad_val = 9
    dummy_dataset = [[1, 2, 3, 1],
                     [1],
                     [1, 2, 3],
                     [1, 2],
                     [1, 2, 3]]
    np_dummy_dataset = np.array(dummy_dataset)

    answer_dataset = [[1, 2, 3],
                      [1, 9, 9],
                      [1, 2, 3],
                      [1, 2, 9],
                      [1, 2, 3]]
    np_answer_dataset = np.array(answer_dataset)

    padded_dataset = pad_sequences(np_dummy_dataset, limit_len, pad_value=pad_val)

    assert padded_dataset.all() == np_answer_dataset.all()
