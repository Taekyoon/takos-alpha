from .tokenization import BertTokenizer


def create_bert_tokenizer(base_path='corpus/vocab.korean.rawtext.list', do_lower_case=False):
    return BertTokenizer.from_pretrained(base_path, do_lower_case=do_lower_case)
