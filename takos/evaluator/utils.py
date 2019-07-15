from takos.utils import load_text

from takos.evaluator.eval_word_segment_model import WordSegmentModelEvaluator


def create_evaluator(type, model, data_builder, dataset_configs, limit_len=None):
    if type == 'word_segment':
        dataset = load_text(dataset_configs['input'])
        input_vocab = data_builder.input_vocab
        tag_vocab = data_builder.tag_vocab

        evaluator = WordSegmentModelEvaluator(model, dataset, input_vocab, tag_vocab, limit_len=limit_len)
    else:
        raise ValueError()

    return evaluator
