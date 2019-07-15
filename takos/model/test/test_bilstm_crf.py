## This code file will be deprecated
import torch

from takos.model.bilstm_crf import BiLSTM_CRF, BiLSTM_CRF_SLU
from takos.model.operations import prepare_sequence
from takos.configs.constants import *

EMBEDDING_DIM = 5
HIDDEN_DIM = 4


def test_run_ner_model_predict():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)

    _, prediction = model(precheck_sent)

    assert prediction.size() == (1, 11)


def test_run_ner_model_train():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tag = prepare_sequence(training_data[0][1], tag_to_ix)

    loss = model.neg_log_likelihood(precheck_sent, precheck_tag)

    assert loss > 0


def test_run_slu_model_predict():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    class_to_ix = {"test_class_1": 0, "test_class_2": 1}

    model = BiLSTM_CRF_SLU(len(word_to_ix), len(class_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)

    _, tag_preds, class_preds = model(precheck_sent)

    assert tag_preds.size() == (1, 11)
    assert class_preds.size() == (1, 2)


def test_run_slu_model_train():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    class_to_ix = {"test_class_1": 0, "test_class_2": 1}

    model = BiLSTM_CRF_SLU(len(word_to_ix), len(class_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tag = prepare_sequence(training_data[0][1], tag_to_ix)
    precheck_class = torch.Tensor([[0]]).long()

    tag_loss, class_loss = model.neg_log_likelihood(precheck_sent, precheck_tag, precheck_class,
                                                    [precheck_sent.size(-1)])

    assert tag_loss + class_loss > 0
