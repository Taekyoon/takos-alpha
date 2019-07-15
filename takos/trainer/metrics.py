import numpy as np

from sklearn.metrics import f1_score, accuracy_score


def f1(pred, target, labels=None):
    if isinstance(pred[0], np.ndarray):
        pred = pred.flatten()
    if isinstance(target[0], np.ndarray):
        target = target.flatten()

    if len(pred) != len(target):
        raise ValueError(
            'both predict and target length is not the same predict is {} and target is {}'.format(len(pred),
                                                                                                   len(target)))

    return f1_score(target, pred, labels=labels, average='weighted')


def acc(pred, target):
    if isinstance(pred[0], np.ndarray):
        pred = pred.flatten()
    if isinstance(target[0], np.ndarray):
        target = target.flatten()

    if len(pred) != len(target):
        raise ValueError(
            'both predict and target length is not the same predict is {} and target is {}'.format(len(pred),
                                                                                                   len(target)))

    return accuracy_score(target, pred)
