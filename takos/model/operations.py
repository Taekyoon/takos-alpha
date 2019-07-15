## This code file will be deprecated
import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, -1)

    return idx


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor([idxs], dtype=torch.long)


def log_sum_exp(vec):
    m = torch.max(vec, -1)[0]
    return m + torch.log(torch.sum(torch.exp(vec - m.unsqueeze(-1)), -1))

    # max_score = vec[0, argmax(vec)]
    # max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    # Compute log sum exp in a numerically stable way for the forward algorithm
