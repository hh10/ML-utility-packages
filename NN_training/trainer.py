import torch
from torch import nn, onnx
from torchvision import transforms
from torch.utils.data import DataLoader

import dataclasses
from enum import Enum
from typing import Callable, Tuple, List, Union, Dict
from collections import defaultdict
from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from datetime import datetime
import glob
import itertools
from functools import partial


class ModelWrapper(nn.Module):
    def __init__(self, model, name, x_shape, num_classes, device, path=None):
        super().__init__()
        self.model = model
        self.name = name
        self.x_shape = x_shape
        self.num_classes = num_classes
        if path is not None:
            self.load(path)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    def save(self, path, also_onnx=True, log=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path + ".pth")
        if log:
            print("Model saved to", path + ".pth")
        if also_onnx:
            self.save_onnx(path, log)

    def save_onnx(self, path, log=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        onnx.export(self.model, torch.ones((2, *self.x_shape)).to(self.device), path + ".onnx", verbose=False, opset_version=16)
        if log:
            print("Model saved to", path + ".onnx")

    def load(self, path):
        self.model.load_state_dict(torch.load(path + ".pth"))


def denormalize(mean, std, x, device):
    means, = torch.tensor(mean).reshape(1, len(mean), 1, 1)
    std_devs = torch.tensor(std).reshape(1, len(std), 1, 1)
    return x * std_devs.to(device) + means.to(device)


@dataclasses.dataclass
class DatasetWrapper():
    name: str
    x_shape: Tuple[int]
    num_classes: int
    mean: Union[None, tuple]
    std: Union[None, tuple]
    normalize: Callable = None
    denormalize: Callable = None

    def __post_init__(self):
        assert (self.mean is None and self.std is None) or (len(self.mean) == len(self.std) and len(self.mean) == self.x_shape[0]), \
            f"x_shape: {self.x_shape}, mean: {self.mean}, std: {self.std}"
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = partial(denormalize, self.mean, self.std)


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

    def add_checkpoint(self, stat_type = None):
        if stat_type is not None:
            self.checkpoints[stat_type].append(self.average)
        else:
            if len(self.checkpoints) == 0:
                self.checkpoints = []
            self.checkpoints.append(self.average)


@dataclasses.dataclass
class Trainer(ABC):
    name: str
    dataset: DatasetWrapper
    model: Union[nn.Module, ModelWrapper]
    loss_criterion: Union[Callable, Dict[str, Callable]]
    optimizer: torch.optim
    num_epochs: int
    lr_scheduler: torch.optim.lr_scheduler = None
    grad_clipping_value: float = 1e-1
    config: Union[dataclasses.dataclass, dict] = None
    loss_stats = defaultdict(StatsCalc)
    acc_stats = defaultdict(StatsCalc)
    stats = defaultdict(StatsCalc)

    def __post_init__(self):
        self.loss_stats = defaultdict(StatsCalc)
        self.acc_stats = defaultdict(StatsCalc)
        self.stats = defaultdict(StatsCalc)

    @abstractmethod
    def loss_and_accuracy(self, device, data, train: bool = True, **kwargs) -> Tuple[torch.tensor, float]:
        """ compute, stores and return loss and accuracy tensors  """
        pass

    def pre_epoch(self, train: bool, epoch: int):
        for sc in {**self.loss_stats, **self.acc_stats}.values():
            sc.reset()
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.stats['lr'].update(self.lr_scheduler.get_last_lr()[0])
                self.stats['lr'].add_checkpoint()
        else:
            self.model.eval()

    def learn(self, device, data, **kwargs):
        loss, _ = self.loss_and_accuracy(device, data, **kwargs)
        loss.backward()
        if self.grad_clipping_value is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clipping_value)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def post_epoch(self, train: bool, epoch: int):
        stats_type = "train" if train else "val"
        for sc in {**self.loss_stats, **self.acc_stats}.values():
            sc.add_checkpoint(stats_type)

    def member_values(self):
        return "\n".join(f"{v.name}: ({v.type.__name__}) = {self.getattr(k)}" for k, v in self.fields.items())


class TrainingRunner():
    def __init__(self, model: ModelWrapper, out_dir: str, trainers: List[Trainer],
                 train_dl: DataLoader, test_dl: DataLoader, save_best_val_acc_model: bool = True):
        self.model = model
        self.trainers = trainers
        self.device = model.device
        self.out_dir = out_dir
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.save_best_val_acc_model = save_best_val_acc_model
        self.name = "_".join([trainer.name for trainer in self.trainers])
        self.last_epoch_accs = None
        self.logger = Logger()

    def pre_epoch(self, train: bool, epoch: int=0):
        for trainer in self.trainers:
            trainer.pre_epoch(train, epoch)

    def predict(self, data, **kwargs):
        for trainer in self.trainers:
            trainer.loss_and_accuracy(self.device, data, train=False, **kwargs)

    def learn(self, data, **kwargs):
        for trainer in self.trainers:
            trainer.learn(self.device, data, **kwargs)

    def post_epoch(self, train: bool, epoch: int=0):
        for trainer in self.trainers:
            trainer.post_epoch(train, epoch)
        if train:
            return
        curr_accs, accs_str = self.get_current_accuracies()
        if self.save_best_val_acc_model and self.last_epoch_accs is not None:
            improved = {k: False for k in  curr_accs.keys()}
            for k, curr_acc in curr_accs.items():
                improved[k] = k in self.last_epoch_accs and self.last_epoch_accs[k] < curr_acc
            improved_keys = [k for k, v in improved.items() if v]
            improved_keys.sort()
            if len(improved_keys) > 0:
                model_path_fmt = lambda model_dir, model_name, l: f"{model_dir}/{model_name}_best_{'_'.join(l)}_valacc_epoch"
                for ks in range(1, len(improved_keys)+1):
                    for ik in itertools.combinations(improved_keys, ks):
                        mp = model_path_fmt(self.out_dir, self.model.name, ik)
                        pms_to_delete = glob.glob(f"{mp}*")
                        for pm in pms_to_delete:
                            os.remove(pm)
                model_path = model_path_fmt(self.out_dir, self.model.name, improved_keys)
                os.makedirs(self.out_dir, exist_ok=True)
                self.model.save(f"{model_path}{epoch+1}", also_onnx=False, log=False)
        self.last_epoch_accs = curr_accs

    def get_current_stats(self, name: str):
        if name == 'losses':
            stats = {k: v.average for trainer in self.trainers for k, v in trainer.loss_stats.items()}
        else:
            stats = {k: v.average for trainer in self.trainers for k, v in trainer.stats[name].items()}
        stats_str = ", ".join([f"{k}: {v:.4f}" for k, v in stats.items()])
        return stats, stats_str

    def get_current_accuracies(self):
        accs = {k: v.average for trainer in self.trainers for k, v in trainer.acc_stats.items()}
        accs_str = ", ".join([f"{k}: {100 * v:.3f}%" for k, v in accs.items()])
        return accs, accs_str

    def train(self, num_epochs: int, save_at_end: bool = False, dummy_run: bool = False):
        # log the trainers to know what exactly was run
        start_time = datetime.now()
        start_time_str = start_time.strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(self.out_dir, f"training_log_{start_time_str}.txt")
        notes_file = os.path.join(self.out_dir, f"notes_{start_time_str}.txt")
        print(f"Training config, progress and results to be logged in {self.out_dir}.")
        self.logger.open(notes_file)
        for trainer in self.trainers:
            self.logger.printf(data_class_str(trainer) + '\n\n\n', notes_file)
        self.logger.open(log_file)

        epoch_pbar = tqdm(range(num_epochs), desc=f"{self.name}: Epochs", leave=True)
        for epoch in epoch_pbar:
            if dummy_run and epoch > 2: break  # noqa: E701
            train_pbar = tqdm(self.train_dl, leave=False)
            test_pbar = tqdm(self.test_dl, leave=False)

            self.pre_epoch(train=True, epoch=epoch)
            for di, data in enumerate(train_pbar):
                if dummy_run and di > 1: break  # noqa: E701
                self.learn(data)
                curr_losses_str, curr_accs_str = self.get_current_stats('losses')[1], self.get_current_accuracies()[1]
                train_pbar.set_description(f"{self.name}: Train Accuracies: {curr_accs_str}")
                self.logger.printf(f"train, {epoch}, " + curr_losses_str + ", " + curr_accs_str + "\n", log_file)
            self.post_epoch(train=True, epoch=epoch)

            self.pre_epoch(train=False, epoch=epoch)
            # with torch.no_grad():
            for di, data in enumerate(test_pbar):
                if dummy_run and di > 1: break  # noqa: E701
                # if di > 19: break  # smaller tests during training?
                self.predict(data)
                test_pbar.set_description(f"{self.name}: Test Accuracies: {self.get_current_accuracies()[1]}")
            self.post_epoch(train=False, epoch=epoch)
            self.logger.printf(f"val, {epoch}, " + self.get_current_stats('losses')[1] + ", " + self.get_current_accuracies()[1] + "\n", notes_file)

            self.plot_progress(self.out_dir, self.name,
                               {f"{trainer.name}_{k}": v for trainer in self.trainers for k, v in trainer.acc_stats.items()},
                               {f"{trainer.name}_{k}": v for trainer in self.trainers for k, v in {**trainer.loss_stats, **trainer.stats}.items()})
            # intermediate model saving
            self.model.save(f"/tmp/{self.model.name}_{start_time_str}", log=False)
        final_accs_str = self.get_current_accuracies()[1]
        self.logger.print(f"Training time: {datetime.now() - start_time}\nFinal test accuracies: {final_accs_str}\nFinal test losses: {self.get_current_stats('losses')[1]}\n\n", notes_file)
        if save_at_end:
            self.model.save(os.path.join(self.out_dir, f"epoch{num_epochs}"))
        self.logger.close()

    def eval(self):
        test_pbar = tqdm(self.test_dl, leave=False)
        self.pre_epoch(train=False)
        # with torch.no_grad():
        for di, data in enumerate(test_pbar):
            self.predict(data)
            test_pbar.set_description(f"{self.name}: Test Accuracies: {self.get_current_accuracies()[1]}")
        self.post_epoch(train=False)

        eval_file = os.path.join(self.out_dir, f"eval_result.txt")
        self.logger.print(f"Evaluation result\ntest_accuracies: {self.get_current_accuracies()[1]}\ntest_losses: {self.get_current_stats('losses')[1]}\n\n", eval_file)

    @staticmethod
    def plot_progress(out_dir, label, accs_stats, stats):
        import numpy as np
        from matplotlib import pyplot as plt
        max_plots_in_chart = 3
        n_acc_cols = int(np.ceil(len(accs_stats) / max_plots_in_chart))
        ncols = n_acc_cols + len(stats)
        fig, axes = plt.subplots(1, ncols, figsize=(10*ncols, 8))

        acc_types = list(accs_stats.keys())
        acc_types.sort()
        for ai, acc_type in enumerate(acc_types):
            if ai % max_plots_in_chart == 0:
                title_str = "ACCS: "
                ax = axes[int(ai / max_plots_in_chart)]
            else:
                title_str += "\n"
            title_str += f"{acc_type}"
            acc_subtypes = list(accs_stats[acc_type].checkpoints.keys())
            acc_subtypes.sort()
            for acc_subtype in acc_subtypes:
                accs = accs_stats[acc_type].checkpoints[acc_subtype]
                ax.plot(accs, label=f"{acc_type} {acc_subtype}")
                title_str += f" {acc_subtype}: {accs[-1]:.2f}"
            ax.set_title(f"{title_str} (epoch: {len(accs)})")
            ax.legend()

        for li, (type_, stats) in enumerate(stats.items()):
            title_str = f"{type_}."
            ax = axes[n_acc_cols + li]
            if isinstance(stats.checkpoints, dict):
                subtypes = list(stats.checkpoints.keys())
                subtypes.sort()
                for subtype in subtypes:
                    vals = stats.checkpoints[subtype]
                    ax.plot(vals, label=subtype)
                    title_str += f"  {subtype}: {vals[-1]:.2f}"
                ax.set_title(f"{title_str} (epoch: {len(vals)})")
            else:
                vals = stats.checkpoints
                ax.plot(vals, label=type)
                ax.set_title(f"{title_str}: {vals[-1]:.2f} (epoch: {len(vals)})")
            ax.legend()

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{label}_training_progress.png"))
        plt.close()


class WarmupScheduleProfile(Enum):
    LINEAR = 'linear'
    EXP = 'exp'
    SSHAPED = 'sshaped'


@dataclasses.dataclass
class WarmupScheduler:
    min_val: float
    max_val: float
    start_epoch: int
    end_epoch: int
    profile: WarmupScheduleProfile
    beta: float = 3

    def __call__(self, epoch: int):
        if epoch <= self.start_epoch:
            return self.min_val
        if epoch >= self.end_epoch:
            return self.max_val

        epoch -= self.start_epoch
        warmup_dur = self.end_epoch - self.start_epoch
        if self.profile is WarmupScheduleProfile.LINEAR:
            return self.min_val + (self.max_val - self.min_val) * epoch / warmup_dur
        elif self.profile is WarmupScheduleProfile.EXP:
            return self.min_val + (self.max_val - self.min_val) / 2 ** (warmup_dur - epoch)
        elif self.profile is WarmupScheduleProfile.SSHAPED:
            e = epoch / warmup_dur
            return self.min_val + (self.max_val - self.min_val) * (e ** self.beta / (e ** self.beta + (1 - e) ** self.beta))
        raise NotImplementedError


def data_class_str(dataclass_instance):
    fields = dataclasses.fields(dataclass_instance)
    return '\n'.join([f"{v.name}: ({v.type}) = {getattr(dataclass_instance, v.name)}" for v in fields])


def set_seeds_and_device(seed: int = 0, use_gpu: bool = True) -> torch.device:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda:0")
    return device


class Logger(object):
    def __init__(self):
        self.handlers = {}

    def open(self, path: str):
        if path not in self.handlers:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.handlers[path] = open(path, 'a+', buffering=1)

    def printf(self, string: str, path: str):
        self.handlers[path].write(string)

    def print(self, string: str, path: str = None):
        print(string)
        if path is not None:
            with open(path, 'a+') as f:
                f.write(string)

    def close(self):
        for path in list(self.handlers.keys()):
            handler = self.handlers.pop(path)
            handler.close()
