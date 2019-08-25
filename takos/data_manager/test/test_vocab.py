from pathlib import Path

from takos.data_manager.vocab import Vocabulary


def test_create_vocabulary():
    dummy_inputs = [['나는', '한국에', '살고', '있어요'],
                    ['한국에', '사는건', '쉽지', '않아요'],
                    ['학교종이', '울리면', '모여야', '해요'],
                    ['학교종이', '울리지', '않으면', '어디로', '가야', '하죠']]
    answer_vocab_size = 20

    vocab = Vocabulary()
    vocab.fit(dummy_inputs)

    sampled_tokens = dummy_inputs[0]
    indices = vocab.to_indices(sampled_tokens)
    reversed_tokens = vocab.to_tokens(indices)

    assert isinstance(indices, list)
    assert isinstance(indices[0], int)
    assert len(vocab) == answer_vocab_size
    assert reversed_tokens == sampled_tokens


def test_unknown_token_to_index():
    dummy_inputs = [['나는', '한국에', '살고', '있어요'],
                    ['한국에', '사는건', '쉽지', '않아요'],
                    ['학교종이', '울리면', '모여야', '해요'],
                    ['학교종이', '울리지', '않으면', '어디로', '가야', '하죠']]
    unknown_token = '기러기'
    unknown_tokens = ['한쿡', '기러기', '정말', '많다']
    unknown_index = 1
    unknown_indices = [1, 1, 1, 1]

    vocab = Vocabulary()
    vocab.fit(dummy_inputs)

    assert unknown_index == vocab.to_indices(unknown_token)
    assert unknown_indices == vocab.to_indices(unknown_tokens)


def test_vocab_obj_as_json():
    json_path = Path('./takos/data_manager/test/test_dataset/vocab_test.json')

    dummy_inputs = [['나는', '한국에', '살고', '있어요'],
                    ['한국에', '사는건', '쉽지', '않아요'],
                    ['학교종이', '울리면', '모여야', '해요'],
                    ['학교종이', '울리지', '않으면', '어디로', '가야', '하죠']]
    vocab, dummy_vocab = Vocabulary(), Vocabulary()
    vocab.fit(dummy_inputs)

    vocab.to_json(json_path)
    dummy_vocab.from_json(json_path)

    assert vocab == dummy_vocab


def test_vocab_obj_index_to_token():
    dummy_inputs = [['나는', '한국에', '살고', '있어요'],
                    ['한국에', '사는건', '쉽지', '않아요'],
                    ['학교종이', '울리면', '모여야', '해요'],
                    ['학교종이', '울리지', '않으면', '어디로', '가야', '하죠']]

    example_tokens = dummy_inputs[0]

    vocab = Vocabulary()
    vocab.fit(dummy_inputs)

    indices = vocab.to_indices(example_tokens)
    target_tokens = vocab.to_tokens(indices)

    assert example_tokens == target_tokens
