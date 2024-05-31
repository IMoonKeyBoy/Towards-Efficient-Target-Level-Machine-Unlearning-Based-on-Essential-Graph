# Copyright (C) 2021-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.
import argparse
import heapq
import json
import random
import os
import numpy as np
import requests
import torch
import torchvision
from cprint import cprint
from pathlib import Path
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tfm, models
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import os
import torch
from cprint import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from dataset.dataset import CelebASingleClassifiation
from dataset.augmentations import get_transforms
from torchcam import methods
from torchcam.utils import overlay_mask
from utils.resnet_cifar import ResNet_CIFAR
from scheme_explanationlayer import explanatorylayer
from scheme_explanatorygraph import explantorygraph
from torchcam import methods
import argparse
import copy
import json
import random
import argparse
import os
import random

import numpy as np
import torch
import wandb
from cprint import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.augmentations import get_transforms
from dataset.dataset import CelebAMultiClassifiation, CelebAMultiClassifiation_evaluate
from functions import get_optimizer, get_scheduler, get_lr, get_multi_classification_model

from pathlib import Path
from scheme_explanatorygraph import explantorygraph
import numpy as np
import requests
import torch
import torchvision
from cprint import *
from torch import nn
from torchvision import models, transforms, datasets
from torchvision import transforms as tfm
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from scheme_explanationlayer import explanatorylayer
from scheme_explanatorygraph import explantorygraph
from utils.data_loader_999 import data_loader
from utils.helper import validate
from utils.resnet_cifar import ResNet_CIFAR

# use the static variables
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_one_epoch(args,
                    epoch,
                    device,
                    model,
                    loader,
                    optimizer,
                    criterion):
    model.train()
    train_loss_list = list()
    train_accuracy = []
    for i in range(len(args.task)):
        train_accuracy.append([])

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        optimizer.zero_grad()
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)

        probs = model(imgs)

        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()

        train_loss = loss.item()
        train_loss_list.append(train_loss)

        for i in range(len(args.task)):
            preds_index = probs[:, i] > 0.6
            preds_acc = (labels[:, i] == preds_index).mean()
            train_accuracy[i].append(preds_acc)

        # refresh the show bar
        desc = f"Train Epoch: {epoch + 1}, Loss: {np.mean(train_loss_list):.2f}"
        for i in range(len(args.task)):
            desc = desc + ", " + args.task[i] + f": {np.mean(train_accuracy[i]):.2f}"
        pbar.set_description(desc)

    return np.mean(train_loss_list), train_accuracy


def valid_one_epoch(args,
                    epoch,
                    device,
                    model,
                    loader,
                    criterion):
    valid_loss_list = list()
    model.eval()

    valid_accuracy = []
    for i in range(len(args.task)):
        valid_accuracy.append([])

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)

        probs = model(imgs)
        torch.set_printoptions(precision=4, sci_mode=False)
        loss = criterion(probs, labels)

        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()

        valid_loss = loss.item()
        valid_loss_list.append(valid_loss)

        for i in range(len(args.task)):
            preds_index = probs[:, i] > 0.6
            preds_acc = (labels[:, i] == preds_index).mean()
            valid_accuracy[i].append(preds_acc)

        # refresh the show bar
        desc = f"Valid Epoch: {epoch + 1}, Loss : {np.mean(valid_loss_list):.2f}"
        for i in range(len(args.task)):
            desc = desc + ", " + args.task[i] + f": {np.mean(valid_accuracy[i]):.2f}"
        pbar.set_description(desc)

    return np.mean(valid_loss_list), valid_accuracy


def main(args):
    args.saved_model_path = os.path.join(args.saved_model_path, args.run_name)
    pretrained_adversary_network_path = os.path.join(args.saved_model_path, args.run_name + "_final.pth")
    if os.path.exists(pretrained_adversary_network_path):
        original_model = torch.load(pretrained_adversary_network_path)
        cprint.warn("Load Checkpoint" + pretrained_adversary_network_path)
    else:
        cprint.warn("No Checkpoint File Found " + pretrained_adversary_network_path)

    train_transforms = get_transforms(args.img_size, mode='train')
    valid_transforms = get_transforms(args.img_size, mode='valid')

    train_dataset = CelebAMultiClassifiation_evaluate(args.data_path, args.task, transforms=train_transforms, is_split_notask_and_task_dataset=True, is_task=True, is_training=True)
    valid_dataset = CelebAMultiClassifiation_evaluate(args.data_path, args.task, transforms=valid_transforms, is_split_notask_and_task_dataset=True, is_task=True, is_training=False)

    train_loader = DataLoader(train_dataset, args.batch_size, drop_last=True, num_workers=6, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, drop_last=False, num_workers=6, pin_memory=True)

    optimizer = get_optimizer(args, original_model)
    scheduler = get_scheduler(args.scheduler, optimizer)
    criterion = torch.nn.BCELoss()

    accuracy_model = copy.deepcopy(original_model)
    with torch.no_grad():
        valid_loss, valid_accuracies = valid_one_epoch(args=args,
                                                       epoch=0,
                                                       device=device,
                                                       model=original_model,
                                                       loader=valid_loader,
                                                       criterion=criterion)

    # for i in range(len(args.task)):
    #    print(args.task[i], np.mean(valid_accuracies[i]))

    myexplanatoryGraph = explantorygraph()
    myexplanatoryGraph.load_target_class(args)
    myexplanatoryGraph.layers = myexplanatoryGraph.layers[-1:]
    # myexplanatoryGraph.print_graph()

    with torch.no_grad():
        accuracy_model = copy.deepcopy(original_model)
        for name, values in accuracy_model.named_parameters():
            for i in range(len(myexplanatoryGraph.layers)):
                layer_name = myexplanatoryGraph.layers[i].name.replace("conv", "batchnorm")
                if layer_name in name or myexplanatoryGraph.layers[i].name in name:
                    for ii in range(len(myexplanatoryGraph.layers[i].nodes_weights)):
                        if myexplanatoryGraph.layers[i].nodes_weights[ii] > float(len(args.task)-1):        # print(myexplanatoryGraph.layers[i].name,myexplanatoryGraph.layers[i].nodes_weights[ii],ii)
                            print(ii,end=", ")
                            mask_values = torch.zeros_like(values[ii], requires_grad=True)
                            values[ii] = torch.nn.Parameter(mask_values)
                    print()
        with torch.no_grad():
            valid_loss, valid_accuracies = valid_one_epoch(args=args,
                                                           epoch=0,
                                                           device=device,
                                                           model=accuracy_model,
                                                           loader=valid_loader,
                                                           criterion=criterion)
            # for i in range(len(args.task)):
            #    print(args.task[i], np.mean(valid_accuracies[i]))


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../datasets/CelebaK/', help='Root path of data')
    parser.add_argument('--saved_model_path', type=str, default='./checkpoints/original_multi/', help='Dir path to save model weights')
    parser.add_argument('--epoch', type=int, default=20, help='Total Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-06, help='Learning Rate')
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))
    parser.add_argument('--optimizer', type=str, default='adamw', choices=('adam', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--scheduler', type=str, default='cosinewarmup', choices=('none', 'cosinewarmup'))
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--training', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--tv_model', type=str, default="resnet18", help='which model you would like to use')
    parser.add_argument('--dataset', type=str, default='celeba', help='dataset?')
    parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for lpython mul   ogging')
    parser.add_argument('--group_name', type=str, default='Multi Classification Training Process', choices=('none', 'cosinewarmup'))

    #parser.add_argument('--task', type=list,  default=['Mouth_Slightly_Open','No_Beard','Eyeglasses'], help='Attributes')
    #parser.add_argument('--task', type=list, default=['Smiling','No_Beard','Eyeglasses'], help='Attributes')
    #parser.add_argument('--task', type=list, default=['Bald','Mouth_Slightly_Open'], help='Attributes')
    #parser.add_argument('--task', type=list, default=['Mouth_Slightly_Open','No_Beard','Wearing_Hat'], help='Attributes')
    parser.add_argument('--task', type=list, default=['Smiling','No_Beard','Wearing_Hat'], help='Attributes')

    parser.add_argument('--target_class', type=int, default=0, help='which image you would like to show')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()
    args.run_name = '_'.join(args.task)
    args.total_class_number = len(args.task)
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    main(args)
