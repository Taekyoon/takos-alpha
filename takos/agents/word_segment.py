import os

import torch
import logging
from pathlib import Path

from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
from eli5.formatters.html import format_as_html

from takos.configs.constants import UNK, RANDOM_SEED

from takos.data_manager.utils import create_builder
from takos.model.utils import create_model
from takos.trainer.utils import create_trainer
from takos.evaluator.utils import create_evaluator

from takos.data_manager.utils import load_vocab_dir
from takos.data_manager.dataset import pad_sequences
from takos.data_manager.tokenizer import SyllableTokenizer

from takos.process.word_segment import segment_word_by_tags

from takos.utils import load_model, load_json, register_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NERExplainerGenerator(object):
    def __init__(self, model, word2idx, max_len):
        self.model = model
        self.word2idx = word2idx
        self.max_len = max_len

    def _preprocess(self, texts):
        if UNK in self.word2idx:
            X = [[self.word2idx.get(w, self.word2idx[UNK]) for w in t.split()]
                 for t in texts]
        else:
            X = [[self.word2idx.get(w, self.word2idx['[UNK]']) for w in t.split()]
                 for t in texts]

        X = pad_sequences(X, self.max_len)
        return torch.Tensor(X).long()

    def get_predict_function(self, word_index):
        def predict_func(texts):
            X = self._preprocess(texts)
            p = self.model.get_probs(X)
            return p[:, word_index, :]

        return predict_func


class Agent(object):
    def _run(self, query: str):
        raise NotImplementedError()

    def __call__(self, query: str):
        return self._run(query)


@register_logging
class WordSegmentAgent(Agent):
    def __init__(self, configs_path):
        self.configs = load_json(configs_path)

        self.task_type = self.configs['type']
        self.tokenizer_type = self.configs['tokenizer'] if 'tokenizer' in self.configs else ''
        self.deploy_path = Path(self.configs['deploy']['path'])
        self.model_configs = self.configs['model']
        self.best_model_path = self.deploy_path / 'model' / 'best_val.pkl'

        if self.tokenizer_type == 'syllable_tokenizer':
            tokenizer = SyllableTokenizer()
        else:
            raise ValueError()

        if not os.path.exists(self.deploy_path / 'dataset'):
            train_dataset_configs = self.configs['dataset']['train']
            create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')

        vocabs = load_vocab_dir(self.task_type, self.deploy_path)

        if 'input_vocab' in vocabs:
            self.model_configs['vocab_size'] = len(vocabs['input_vocab'].word_to_idx)

        if 'label_vocab' in vocabs:
            tag_vocab = vocabs['label_vocab']

        model = create_model(self.task_type, tag_vocab, self.model_configs)
        if os.path.exists(self.best_model_path):
            model = load_model(self.best_model_path, model)
        model.eval()

        self.tokenizer = tokenizer
        self.vocab = vocabs['input_vocab']
        self.label = vocabs['label_vocab']
        self.model = model
        self.preprocess = lambda x: x
        self.postprocess = segment_word_by_tags

    def _run(self, query: str):
        prepro_query = self.preprocess(query)
        tokenized_query = self.tokenizer.tokenize(prepro_query).split()
        indiced_query = self.vocab.to_indices(tokenized_query)

        model_inputs = torch.Tensor([indiced_query]).long()

        pred_score, tag_seq = self.model(model_inputs)
        labeled_tag_seq = self.label.to_tokens(tag_seq[0].tolist())

        post_processed = self.postprocess(prepro_query, labeled_tag_seq)

        pred_score = pred_score.detach().numpy()

        outputs = {'input': prepro_query,
                   'label': labeled_tag_seq,
                   'sequence_score': pred_score,
                   'output': post_processed,
                   'segment_pos': [i for i, x in enumerate(labeled_tag_seq) if x == 'B' or x == 'S']
                   }

        return outputs

    def train(self):
        train_configs = self.configs['train']
        train_dataset_configs = self.configs['dataset']['train']
        gpu_device = self.configs['gpu_device']

        data_builder = create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')

        if data_builder.word_to_idx:
            self.model_configs['vocab_size'] = len(data_builder.word_to_idx)

        if data_builder.tag_to_idx:
            tag_to_idx = data_builder.tag_to_idx

        model = create_model(self.task_type, tag_to_idx, self.model_configs)
        if 'load_model' in self.configs:
            logger.info('load model: {}'.format(self.configs['load_model']))
            if 'load_model_strict' in self.configs:
                strict = self.configs['load_model_strict']
            else:
                strict = False
            logger.info('set load model as strict method: {}'.format(strict))
            if 'load_model' in self.configs and self.configs['load_model'] is not None:
                model = load_model(self.configs['load_model'], model, strict=strict)
        trainer = create_trainer(self.task_type, model, data_builder, train_configs,
                                 gpu_device=gpu_device, deploy_path=self.deploy_path / 'model')

        logger.info(model)
        trainer.train()

    def eval(self):
        test_dataset_configs = self.configs['dataset']['test'] if 'test' in self.configs['dataset'] else None
        train_dataset_configs = self.configs['dataset']['train']

        limit_len = test_dataset_configs['limit_len'] if 'limit_len' in test_dataset_configs else None

        data_builder = create_builder(self.task_type, train_dataset_configs, deploy_path=self.deploy_path / 'dataset')
        test_dataset_configs = self.configs['dataset']['test'] if 'test' in self.configs['dataset'] else None
        evaluator = create_evaluator(self.task_type, self.model, data_builder, test_dataset_configs,
                                     limit_len)
        evaluator.eval()
        logger.info(evaluator.summary())

    def _lime_analyze(self, query, indicies, max_len, max_replace, top_targets=None):
        model = self.model
        vocab = self.vocab.word_to_idx
        label = self.label.word_to_idx
        prepro_query = self.preprocess(query)

        explainer_generator = NERExplainerGenerator(model, vocab, max_len)

        sampler = MaskingTextSampler(
            replacement=UNK,
            max_replace=max_replace,
            token_pattern=None,
            bow=False
        )

        explainer_list = list()
        for i in indicies:
            predict_fn = explainer_generator.get_predict_function(i)

            te = TextExplainer(
                sampler=sampler,
                position_dependent=True,
                random_state=RANDOM_SEED,
            )

            te.fit(' '.join(prepro_query), predict_fn)

            pred_explain = te.explain_prediction(target_names=[l for l in label][3:], top_targets=top_targets)
            explainer_list.append(pred_explain)

        return explainer_list

    def get_lime_analyze_as_html(self, query, indicies, max_len=50, max_replace=0.7, top_targets=None):
        query = query.replace(' ', '')

        for i in indicies:
            if i >= len(query):
                raise ValueError()

        def build_query_info_html(query, index):
            _html_format = ''
            _html_format += '<p>' + 'query: ' + query + '</p><br>'
            _html_format += '<p>' + 'index: ' + str(index) + '</p><br>'
            _html_format += '<p>' + 'char: ' + query[index] + '</p><br>'

            return _html_format

        explainers = self._lime_analyze(query, indicies, max_len, max_replace, top_targets=top_targets)
        html_format = ''
        html_format += '<div>'

        for i, e in zip(indicies, explainers):
            html_format += '<table class="table"><tr>'
            html_format += '<td><div>' + build_query_info_html(query, i) + '</div></td>'
            html_format += '<td><div>' + format_as_html(e).replace('\n', '') + '</div></td>'
            html_format += '</tr></table>'
            html_format += '<hr>'

        html_format += '</div>'

        return html_format

    def get_lime_analyze_as_object(self, query, indicies, max_len=50, max_replace=0.7, top_targets=None):
        query = query.replace(' ', '')
        explainers = self._lime_analyze(query, indicies, max_len, max_replace, top_targets=top_targets)

        return explainers
