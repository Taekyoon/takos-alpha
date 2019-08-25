from takos.data_manager.tokenizer import SyllableTokenizer


def test_syllable_tokenizer():
    dummy_single_sent = "오늘 날씨가 매우 맑습니다"
    dummy_multi_sent = ["오늘 날씨는 어떻습니까",
                        "내일 날씨도 좋을까요",
                        "하루가 너무 좋습니다"]

    eumjeol_tokenizer = SyllableTokenizer()

    tokenized_single_sent = eumjeol_tokenizer.tokenize(dummy_single_sent)
    tokenized_multi_sent = eumjeol_tokenizer.tokenize(dummy_multi_sent)

    assert isinstance(tokenized_single_sent, str)
    assert isinstance(tokenized_multi_sent, list)
