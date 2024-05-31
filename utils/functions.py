import random
from typing import Optional
import numpy as np
import torch.nn as nn
from termcolor import cprint
from torch import Tensor
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torch

# use the static variables
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
    """CAM normalization."""
    # cprint("_CAM._normalize", color="red")
    spatial_dims = cams.ndim - 1 if spatial_dims is None else spatial_dims
    cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
    cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])
    return cams

def generate_test_unlearning_dataset(dataset, list_classes_unlearning: list, list_classes_remaining: list, notice: str):
    cprint(notice, color="red")
    '''
    labels = []
    for label_id in list_classes:
        labels.append(list(dataset.classes)[int(label_id)])
    '''

    sub_dataset_unlearning = []
    sub_dataset_remaining = []
    for datapoint in tqdm(dataset):
        _, label_index = datapoint  # Extract label
        if label_index in list_classes_unlearning:
            sub_dataset_unlearning.append(datapoint)
        if label_index in list_classes_remaining:
            sub_dataset_remaining.append(datapoint)
    return sub_dataset_unlearning, sub_dataset_remaining

def generate_one_samples(dataset, list_classes: list):
    sub_dataset = []
    for datapoint in tqdm(dataset):
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            list_classes.remove(label_index)
            sub_dataset.append(datapoint)
    return sub_dataset


def model_test(net, testloader, notice: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cprint(notice, color="blue")
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # for i in range(len(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    val_acc = correct / total
    return val_acc


# Run data generation
def generate_cifar10(dataset, list_classes: list):
    labels = []
    for label_id in list_classes:
        labels.append(list(dataset.classes)[int(label_id)])

    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            sub_dataset.append(datapoint)
    return sub_dataset


def testmodel(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    # print(val_acc)
    loss = test_loss / num_val_steps
    return loss, val_acc
