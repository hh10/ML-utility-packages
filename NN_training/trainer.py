import torch
from torch import nn

import dataclasses
from typing import Callable, Tuple, List, Union
from collections import defaultdict
from tqdm import tqdm
import os
from abc import ABC, abstractmethod


class StatsCalc(object):
    def __init__(self):
        self.average = 0
        self.checkpoints = defaultdict(list)
        self.reset()

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0.
        self.last_average = self.average
        self.average = 0.

    def update(self, value, weight=1):
        self.value = value
        self.sum += value * weight
        self.count += weight
        self.average = self.sum / self.count

    def add_checkpoint(self, stat_type):
        self.checkpoints[stat_type].append(self.average)


@dataclasses.dataclass
class Trainer(ABC):
    name: str
    loss_criterion: Callable
    model: nn.Module
    optimizer: torch.optim
    lr_scheduler: torch.optim.lr_scheduler = None
    loss_stats = defaultdict(StatsCalc)
    acc_stats = defaultdict(StatsCalc)

    @abstractmethod
    def loss_and_accuracy(self, device, data, **kwargs) -> Tuple[torch.tensor, float]:
        """ compute, stores and return loss and accuracy tensors  """
        pass

    def pre_epoch(self, train: bool):
        for sc in {**self.loss_stats, **self.acc_stats}.values():
            sc.reset()
        self.optimizer.zero_grad()
        if train:
            self.model.train()
        else:
            self.model.eval()

    def learn(self, device, data, **kwargs):
        loss, _ = self.loss_and_accuracy(device, data, **kwargs)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def post_epoch(self, train: bool):
        stats_type = "train" if train else "val"
        for sc in {**self.loss_stats, **self.acc_stats}.values():
            sc.add_checkpoint(stats_type)

    def member_values(self):
        return "\n".join(f"{v.name}: ({v.type.__name__}) = {self.getattr(k)}" for k, v in self.fields.items())


class TrainingRunner():
    def __init__(self, trainers: List[Trainer], device: Union[str, torch.device]):
        self.trainers = trainers
        self.device = device

    def pre_epoch(self, train: bool):
        for trainer in self.trainers:
            trainer.pre_epoch(train)

    def predict(self, data, **kwargs):
        for trainer in self.trainers:
            trainer.loss_and_accuracy(self.device, data, **kwargs)

    def learn(self, data, **kwargs):
        for trainer in self.trainers:
            trainer.learn(self.device, data, **kwargs)

    def post_epoch(self, train: bool):
        for trainer in self.trainers:
            trainer.post_epoch(train)

    def get_current_accuracies(self):
        return {k: v.average for trainer in self.trainers for k, v in trainer.acc_stats.items()}

    def train(self, num_epochs, train_dl, val_dl, out_dir, dummy_run=False):
        # log the trainers to know what happened
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "training_config.txt"), 'a+') as tcf:
            for trainer in self.trainers:
                tcf.write(data_class_str(trainer) + '\n')

        accs_str = lambda: ", ".join([f"{k}: {100 * v:.3f}%" for k, v in self.get_current_accuracies().items()])  # noqa: E731
        training_title = " ".join([trainer.name for trainer in self.trainers])
        epoch_pbar = tqdm(range(num_epochs), desc=f"{training_title}: Epochs", leave=True)
        for epoch in epoch_pbar:
            if dummy_run and epoch > 1: break  # noqa: E701
            train_pbar = tqdm(train_dl, leave=False)
            test_pbar = tqdm(val_dl, leave=False)

            self.pre_epoch(train=True)
            for di, data in enumerate(train_pbar):
                if dummy_run and di > 3: break  # noqa: E701
                self.learn(data)
                train_pbar.set_description(f"{training_title}: Train Accuracies: {accs_str()}")
            self.post_epoch(train=True)

            self.pre_epoch(train=False)
            with torch.no_grad():
                for di, (x, _label) in enumerate(test_pbar):
                    if dummy_run and di > 3: break  # noqa: E701
                    self.predict(data)
                    test_pbar.set_description(f"{training_title}: Test Accuracies: {accs_str()}")
            self.post_epoch(train=False)

            self.plot_progress(out_dir, training_title, 
                               {k: v for trainer in self.trainers for k, v in trainer.loss_stats.items()},
                               {k: v for trainer in self.trainers for k, v in trainer.acc_stats.items()})
        print(f"Final test {training_title} accuracies: {accs_str()}")
        print(f"Training config, progress and results logged in {out_dir}.")

    @staticmethod
    def plot_progress(out_dir, label, losses_stats, accs_stats):
        from matplotlib import pyplot as plt
        ncols = 1 + len(losses_stats)
        fig, ax = plt.subplots(1, ncols, figsize=(10*ncols, 8))
        for li, (loss_type, loss_stats) in enumerate(losses_stats.items()):
            title_str = f"{loss_type}."
            for loss_subtype, losses in loss_stats.checkpoints.items():
                ax[li].plot(losses, label=loss_subtype)
                title_str += f"  {loss_subtype}: {losses[-1]:.2f}"
            ax[li].set_title(f"{title_str} (epoch: {len(losses)}")
            ax[li].legend()
        title_str = ""
        for ai, (acc_type, acc_stats) in enumerate(accs_stats.items()):
            title_str += f"{acc_type}."
            for acc_subtype, accs in acc_stats.checkpoints.items():
                ax[-1].plot(accs, label=acc_subtype)
                title_str += f"  {acc_subtype}: {accs[-1]:.2f}"
            ax[-1].set_title(f"{title_str} (epoch: {len(accs)})")
            ax[-1].legend()

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{label}_training_progress.png"))
        plt.close()


def data_class_str(dataclass_instance):
    fields = dataclasses.fields(dataclass_instance)
    return '\n'.join([f"{v.name}: ({v.type}) = {getattr(dataclass_instance, v.name)}" for v in fields])
