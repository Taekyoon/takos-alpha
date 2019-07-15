from takos.configs.constants import RANDOM_SEED

import gc

import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path

from takos.trainer.trainer import Trainer
from takos.trainer.metrics import f1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SequenceTaggingModelTrainer(Trainer):
    def __init__(self,
                 train_data_loader,
                 valid_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 deploy_path=Path('./tmp'),
                 learning_rate=3e-4,
                 optimizer=torch.optim.Adam,
                 metric_fn=None,
                 gpu_device=-1,
                 random_seed=RANDOM_SEED,
                 eval_labels=None):

        self._epochs = epochs
        self._eval_steps = eval_steps
        self._learning_rate = learning_rate
        self._random_seed = random_seed

        self._deploy_path = deploy_path

        self._train_data_loader = train_data_loader
        self._valid_data_loader = valid_data_loader
        self._model = model
        self._optimizer = optimizer(params=self._model.parameters(), lr=self._learning_rate)

        self._device = torch.device('cuda:' + str(gpu_device)) if torch.cuda.is_available() \
                                                                  and gpu_device >= 0 else torch.device('cpu')
        self._tb_writer = SummaryWriter(self._deploy_path / 'logs')

        if self._device.type == 'cpu':
            torch.manual_seed(self._random_seed)
        else:
            torch.cuda.manual_seed_all(self._random_seed)

        self.train_loss = -1
        self.best_val_f1_score = 0.

        self._metric_fn = metric_fn

        self._eval_labels = eval_labels

        logger.info('deploy path: {}'.format(self._deploy_path))
        logger.info('random seed number: {}'.format(self._random_seed))
        logger.info('learning rate: {}'.format(self._learning_rate))
        logger.info('evaluation check steps: {}'.format(self._eval_steps))
        logger.info('number of epochs: {}'.format(self._epochs))
        logger.info('training device: {}'.format(self._device.type))
        logger.info('eval labels: {}'.format(self._eval_labels))

    def _eval(self):
        gc.collect()

        score, f1_score = 0., 0.

        logger.info('now evaluating...')

        self._model.eval()

        score_failure_cnt = 0

        for step, sampled_batch in tqdm(enumerate(self._valid_data_loader), desc='valid steps',
                                        total=len(self._valid_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            pred_score, tag_seq = self._model(input_batch)
            score += torch.mean(pred_score).item()
            try:
                f1_score += f1(tag_seq.cpu().detach().numpy(), target_batch.cpu().detach().numpy(),
                               labels=self._eval_labels)
            except Exception as e:
                score_failure_cnt += 1
                logger.error("Error message while calculating f1 score: {}".format(e))
                logger.info('f1 score failure {} times'.format(score_failure_cnt))
        else:
            score = score / (step + 1)
            f1_score = f1_score / (step + 1 - score_failure_cnt)

        return score, f1_score

    def _train_epoch(self, epoch):
        steps_in_epoch = len(self._train_data_loader)
        tr_loss = 0.

        logger.info('start training epoch {}'.format(epoch + 1))

        self._model.to(self._device)
        self._model.train()

        for step, sampled_batch in tqdm(enumerate(self._train_data_loader), desc='train steps',
                                        total=len(self._train_data_loader)):
            batch_size = sampled_batch['inputs']['value'].size(0)
            total_steps = epoch * steps_in_epoch + (step + 1)

            input_batch = sampled_batch['inputs']['value'].view(batch_size, -1).to(self._device)
            target_batch = sampled_batch['entities'].view(batch_size, -1).to(self._device)

            loss = self._model.loss(input_batch, target_batch)
            tr_loss += loss.cpu().item()
            self._backprop(loss, self._optimizer)

            if self._eval_steps > 0 and total_steps % self._eval_steps == 0:
                val_score, val_f1 = self._eval()

                if val_f1 > self.best_val_f1_score:
                    self.best_val_f1_score = val_f1

                    self._save_model(self._model, self._deploy_path / 'best_val.pkl')

                self._model.train()
                logger.info(
                    'epoch : {}, steps : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                                                                                                           total_steps,
                                                                                                           tr_loss / (step + 1),
                                                                                                           val_f1,
                                                                                                           val_score))
                self._tb_writer.add_scalar('train_loss', tr_loss / (step + 1), total_steps)
                self._tb_writer.add_scalar('train_tag_loss', tr_loss / (step + 1), total_steps)

                self._tb_writer.add_scalar('valid_tag_f1', val_f1, total_steps)

                filename = 'checkpoint_' + str(total_steps) + '_model.pkl'
                self._save_model(self._model, self._deploy_path / 'checkpoint' / filename)
                gc.collect()
        else:
            logger.info('epoch {} is done!'.format(epoch + 1))
            val_score, val_f1 = self._eval()

            if val_f1 > self.best_val_f1_score:
                self.best_val_f1_score = val_f1

                self._save_model(self._model, self._deploy_path / 'best_val.pkl')

            self._model.train()
            logger.info(
                'epoch : {}, steps : {}, tr_loss : {:.3f}, val_f1 : {:.3f}, val_score : {:.3f}'.format(epoch + 1,
                                                                                                       total_steps,
                                                                                                       tr_loss / (step + 1),
                                                                                                       val_f1,
                                                                                                       val_score))
        logger.info('current best tag f1: {:.3f}'.format(self.best_val_f1_score))
        gc.collect()
        return tr_loss / (step + 1)
