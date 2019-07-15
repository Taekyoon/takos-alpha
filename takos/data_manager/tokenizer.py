try:
    from konlpy.tag import Komoran, Mecab
except:
    pass

from typing import List


class Tokenizer(object):
    def __init__(self, tokenize_model=None):
        self.model = tokenize_model

    def tokenize(self, sents: List[str] or str) -> List[str] or str:
        to_reduce = False
        if isinstance(sents, str):
            tokenized_sents = self._operate(sents)
            tokenized_sents = [tokenized_sents]
            to_reduce = True

        if isinstance(sents, list):
            tokenized_sents = [self._operate(s) for s in sents]

        return tokenized_sents[0] if to_reduce else tokenized_sents

    def _operate(self, sent: str) -> str:
        pass


class SyllableTokenizer(Tokenizer):
    def __init__(self):
        super(SyllableTokenizer, self).__init__()

    def _operate(self, sent: str):
        unspaced_sent = sent.replace(' ', '')

        return ' '.join(unspaced_sent).rstrip()


class KomoranTokenizer(Tokenizer):
    def __init__(self):
        super(KomoranTokenizer, self).__init__(Komoran())

    def _operate(self, sent: str) -> str:
        tokenized_list = self.model.morphs(sent)

        return ' '.join(tokenized_list).rstrip()


class MecabTokenizer(Tokenizer):
    def __init__(self):
        super(MecabTokenizer, self).__init__(Mecab())

    def _operate(self, sent: str) -> str:
        tokenized_list = self.model.morphs(sent)

        return ' '.join(tokenized_list).rstrip()
