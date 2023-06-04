import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import random
import numpy as np
from typing import Tuple

from trainer import DatasetWrapper


def set_seeds_and_device(seed: int = 0, use_gpu: bool = True) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda:0")
    return device


def get_CIFAR10_dataloaders(dataset_root: str, x_shape) -> Tuple[DatasetWrapper, DataLoader, DataLoader]:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)
    train_dl = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
    test_dl = DataLoader(testset, batch_size=100, shuffle=False, num_workers=16, pin_memory=True)
    dataset = DatasetWrapper(name='CIFAR10', x_shape=x_shape, num_classes=10, mean=mean, std=std)
    return dataset, train_dl, test_dl


def get_tiny_resnet18(num_classes: int) -> nn.Sequential:
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    resnet_layers = list(resnet.children())
    del resnet_layers[3]
    last_layer_in_features = resnet_layers[-1].in_features
    resnet_layers = resnet_layers[:-1]
    resnet_layers.append(nn.Flatten())
    resnet_layers.append(nn.Linear(last_layer_in_features, num_classes))
    return nn.Sequential(*resnet_layers)
