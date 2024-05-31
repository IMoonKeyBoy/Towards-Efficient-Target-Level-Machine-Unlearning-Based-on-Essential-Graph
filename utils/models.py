# **************
# MINST model:
# **************
import torch
from torch import nn
import torch.nn.functional as F


class second_AlexNet_cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(second_AlexNet_cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

class second_AlexNet_cifar100(nn.Module):
    def __init__(self, num_classes=100):
        super(second_AlexNet_cifar100, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x