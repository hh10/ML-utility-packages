import torch
from torch import nn

from examples_utils import set_seeds_and_device, get_CIFAR10_dataloaders, get_tiny_resnet18
from trainer import ModelWrapper, Trainer, TrainingRunner

dataset_root = "./../datasets/cifar"

device = set_seeds_and_device()
dataset, train_dl, test_dl = get_CIFAR10_dataloaders(dataset_root, (3, 32, 32))
tresnet18 = get_tiny_resnet18(num_classes=dataset.num_classes).to(device)
model = ModelWrapper(tresnet18, 'tiny_resnet18', dataset.x_shape, dataset.num_classes, device)


class ClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__("Classification", *args, **kwargs)

    def loss_and_accuracy(self, device, data, *args, **kwargs):
        x, y = data[0].to(device), data[1].to(device)
        logits = self.model(x)
        loss = self.loss_criterion(logits, y)
        self.loss_stats["CEloss"].update(loss.item())
        acc = (logits.argmax(-1) == y).float().mean().item()
        self.acc_stats["Cla"].update(acc)
        return loss, acc


num_epochs = 10
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tresnet18.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dl), eta_min=1e-4)
tresnet18_cla_trainer = ClassificationTrainer(dataset, model, loss_criterion, optimizer, num_epochs, lr_scheduler)

out_dir = "/tmp/trainer_class_classifier_example"
training_runner = TrainingRunner(model, out_dir, [tresnet18_cla_trainer], train_dl, test_dl)
training_runner.train(num_epochs, dummy_run=True)
