import argparse
import os
import sys
import time
import numpy as np
import random
import torch
import torch.nn as nn
from cprint import *
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from torchvision import models
from collections import OrderedDict
import torch.nn as nn
import wandb


import os
import sys

from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #print(correct.shape)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])
    )



    list_allclasses = list(range(1000))
    list_allclasses.remove(unlearning_class)  # rest classes
    rest_traindata = generate(train_dataset, list_allclasses)
    rest_trainloader = torch.utils.data.DataLoader(rest_traindata, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=pin_memory,
                                                   sampler=None)

    unlearn_listclass = [unlearning_class]
    unlearning_traindata = generate(train_dataset, unlearn_listclass)
    unlearning_trainloader = torch.utils.data.DataLoader(unlearning_traindata, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=pin_memory,
                                                   sampler=None)

    rest_testdata = generate(val_dataset, list_allclasses)
    unlearn_testdata = generate(val_dataset, unlearn_listclass)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=pin_memory)
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=batch_size, shuffle=False,
                                                     num_workers=workers, pin_memory=pin_memory)

    return rest_trainloader, unlearning_trainloader, rest_testloader, unlearn_testloader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="", help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', default=False, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='/data/hengxu/PycharmWorkspace/Technical Paper Two/ImageNet_training/49_resnet18_2022-10-06 16-07-09.pth', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('-u', '--unlearning_class', default= 0,
                    help='evaluate model on validation set')
best_prec1 = 0.0


def main():
    # use the static variables
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, best_prec1
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    # Model selection
    if args.arch is not None:
        model = models.__dict__[args.arch]()
    #print(model)

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    args.data = Path(__file__).resolve().home() / "Data/PycharmWorkspace/datasets/ImageNet/"
    rest_trainloader, unlearning_trainloader, rest_testloader, unlearn_testloader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory, args.unlearning_class)

    prec1, prec5 = validate(rest_testloader, model, criterion, args.print_freq)
    print(prec1)
    prec1, prec5 = validate(unlearning_trainloader, model, criterion, args.print_freq)
    print(prec1)


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_remaining = AverageMeter()
    top5_remaining = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1_remaining.update(prec1[0], input.size(0))
            top5_remaining.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return top1_remaining.avg, top5_remaining.avg


if __name__ == '__main__':
    main()
