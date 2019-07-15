import torch
from torch import nn

import jiwer
from tqdm import tqdm
from typing import List

import logging
from prettytable import PrettyTable

from takos.data_manager.vocab import Vocabulary
from takos.process.common import unspacing, text_to_list
from takos.process.word_segment import segment_word_by_tags, labelize

from takos.trainer.metrics import f1, acc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WordSegmentModelEvaluator(object):
    def __init__(self,
                 model: nn.Module,
                 dataset: List,
                 input_vocab: Vocabulary,
                 tag_vocab: Vocabulary,
                 limit_len=None):
        self._model = model
        self._input_vocab = input_vocab
        self._tag_vocab = tag_vocab

        self.limit_len = limit_len

        self._dataset = dataset

        self._device = torch.device('cpu')

        self._wer_score = None
        self._corrected_sent_cnt = None
        self._acc_score = None
        self._f1_score = None

    def eval(self):
        logger.info('now evaluate!')

        self._model.eval()

        wer_score = 0.
        acc_score = 0.
        f1_score = 0.
        corrected_sent_cnt = 0.

        score_failure_cnt = 0

        for step, text in tqdm(enumerate(self._dataset), desc='evaluation steps', total=len(self._dataset)):
            if self.limit_len is not None:
                text = text[:self.limit_len]
            try:
                unspaced_text = unspacing(text.strip())
                tokenized_text = text_to_list(unspaced_text)

                input_batch = torch.Tensor([self._input_vocab.to_indices(tokenized_text)]).long()

                _, tag_seq = self._model(input_batch)
                labeled_tag_seq = self._tag_vocab.to_tokens(tag_seq[0].tolist())
                pred_text = segment_word_by_tags(unspaced_text, labeled_tag_seq)
                wer_score += jiwer.wer(text.strip(), pred_text.strip())
                if text.split() == pred_text.split():
                    corrected_sent_cnt += 1

                _, labels = labelize(text, bi_tags_only=True)
                labels = [ch for ch in labels]
                labeled_tag_seq = ' '.join(labeled_tag_seq).replace('E', 'I').replace('S', 'B').replace('<pad>',
                                                                                                        'I').split()
                acc_score += acc(labeled_tag_seq, labels)
                f1_score += f1(labeled_tag_seq, labels, labels=['B', 'I'])
            except Exception as e:
                score_failure_cnt += 1
                logger.warning("Error message while calculating wer score: {}".format(e))
                logger.info('wer score failure {} times'.format(score_failure_cnt))
                raise ValueError()

        else:
            wer_score = wer_score / (step + 1 - score_failure_cnt)
            corrected_sent_cnt = corrected_sent_cnt / (step + 1 - score_failure_cnt)
            acc_score = acc_score / (step + 1 - score_failure_cnt)
            f1_score = f1_score / (step + 1 - score_failure_cnt)

        self._wer_score = wer_score
        self._corrected_sent_cnt = corrected_sent_cnt
        self._acc_score = acc_score
        self._f1_score = f1_score

        logger.info('evaluation done!')

    def summary(self):
        table = PrettyTable(['Name', 'Score'])
        table.add_row(['WER score', "{:.4f}".format(self._wer_score)])
        table.add_row(['SER score', "{:.4f}".format(1. - self._corrected_sent_cnt)])
        table.add_row(['F1 score', "{:.4f}".format(self._f1_score)])
        table.add_row(['ACC score', "{:.4f}".format(self._acc_score)])

        return table
