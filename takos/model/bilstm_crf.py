## This code file will be deprecated

import torch
from torch import nn

from takos.configs.constants import START_TAG, STOP_TAG
from takos.model.operations import argmax, log_sum_exp


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout=0.5, num_layers=1):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.num_layers = num_layers

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=num_layers, bidirectional=True, dropout=dropout)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.device = torch.device('cpu')

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        torch.nn.init.xavier_uniform_(self.hidden2tag.weight)
        # self.hidden = self.init_hidden()

    def to(self, device):
        super(BiLSTM_CRF, self).to(device)
        self.device = device

        return

    def init_hidden(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).to(self.device),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2).to(self.device))

    def _forward_alg(self, feats):
        batch_size = feats.size(0)
        sequence_length = feats.size(1)
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        trans = self.transitions.unsqueeze(0)

        for t in range(sequence_length):
            emit_score = feats[:, t].unsqueeze(2)
            next_tag_var = forward_var.unsqueeze(1) + emit_score + trans
            forward_var = log_sum_exp(next_tag_var)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, lens=None):
        batch_size = sentence.size(0)
        self.hidden = self.init_hidden(batch_size)

        embeds = self.word_embeds(sentence)

        lstm_out, _ = self.lstm(embeds, self.hidden)
        if lens is not None:
            self.hidden = torch.cat([lstm_out[i, l-1, :] for i, l in enumerate(lens)]).view(batch_size, -1)
        else:
            self.hidden = lstm_out[:, -1, :]
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_batch_sentences(self, batch_feats, batch_tags):
        batch_scores = []
        for f, t in zip(batch_feats, batch_tags):
            batch_scores.append(self._score_sentence(f, t))

        return torch.cat(batch_scores)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()

        best_path = torch.cat(best_path)

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, lens=None):
        feats = self._get_lstm_features(sentence, lens=lens)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_batch_sentences(feats, tags)

        return torch.mean(forward_score - gold_score)

    def forward(self, sentences):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentences)

        scores, tag_seqs = [], []
        # Find the best path, given the features.
        for l_f in lstm_feats:
            score, tag_seq = self._viterbi_decode(l_f)
            scores.append(score)
            tag_seqs.append(tag_seq)

        scores = torch.stack(scores)
        tag_seqs = torch.stack(tag_seqs)

        return scores, tag_seqs


class BiLSTM_CRF_SLU(BiLSTM_CRF):
    def __init__(self, vocab_size, class_size, tag_to_ix, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(BiLSTM_CRF_SLU, self).__init__(vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                                             num_layers=num_layers, dropout=dropout)

        self.class_size = class_size

        self.hidden2class = nn.Linear(self.hidden_dim, self.class_size)
        self.ce_loss = nn.CrossEntropyLoss()

        torch.nn.init.xavier_uniform_(self.hidden2class.weight)

    def _class_features(self):
        class_feats = self.hidden2class(self.hidden)

        return class_feats

    def neg_log_likelihood(self, sentence, tags, classes, lens):
        tag_loss = super(BiLSTM_CRF_SLU, self).neg_log_likelihood(sentence, tags, lens=lens)
        class_feats = self._class_features()
        class_loss = self.ce_loss(class_feats, classes.squeeze(-1))

        return tag_loss, class_loss

    def forward(self, sentences):
        scores, tag_seqs = super(BiLSTM_CRF_SLU, self).forward(sentences)
        class_feats = self._class_features()
        class_probs = nn.functional.softmax(class_feats, dim=-1)

        return scores, tag_seqs, class_probs
