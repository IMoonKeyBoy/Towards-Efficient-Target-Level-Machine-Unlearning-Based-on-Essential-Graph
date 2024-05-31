import os
import sys

from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_index1(lst=None, item=[]):
    return [index for (index, value) in enumerate(lst) if value in item]


def generate(dataset, list_classes: list):
    '''
    sub_dataset = []
    for datapoint in tqdm(dataset):
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)
    '''
    return Subset(dataset, get_index1(dataset.targets, list_classes))


def data_loader(root, batch_size=256, workers=1, pin_memory=True, unlearning_class = None):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    list_allclasses = list(range(1000))
    list_allclasses.remove(unlearning_class)  # rest classes
    rest_traindata = generate(train_dataset, list_allclasses)
    rest_trainloader = torch.utils.data.DataLoader(rest_traindata, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=pin_memory,
                                                   sampler=None)

    unlearn_listclass = [unlearning_class]
    rest_testdata = generate(val_dataset, list_allclasses)
    unlearn_testdata = generate(val_dataset, unlearn_listclass)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=pin_memory)
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=batch_size, shuffle=False,
                                                     num_workers=workers, pin_memory=pin_memory)

    return rest_trainloader, rest_testloader, unlearn_testloader