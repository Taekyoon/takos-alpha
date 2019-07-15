import torch

from pathlib import Path

from takos.utils import make_dir_if_not_exist


class Trainer(object):
    def train(self):
        torch.manual_seed(self._random_seed)
        for i in range(self._epochs):
            self.train_loss = self._train_epoch(i)

    def _backprop(self, loss: torch.nn.Module, optimizer:torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_model(self, model: torch.nn.Module, path: Path):
        make_dir_if_not_exist(path.parents[0])
        torch.save(model.state_dict(), path)

    def _train_epoch(self, epoch: int):
        raise NotImplementedError()
