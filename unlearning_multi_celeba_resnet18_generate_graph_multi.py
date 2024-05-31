import argparse
import heapq
import json
import random
import os
import time
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
from dataset.dataset import CelebAMultiClassifiation,CelebAMultiClassifiation_evaluate
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

def evaluate_cam_classs_idx(args, temp_target_class, model):
    valid_transforms = get_transforms(args.img_size, mode='valid')
    valid_dataset = CelebASingleClassifiation(args.data_path, task_labels=[args.task_labels[temp_target_class]], transforms=valid_transforms, is_split_notask_and_task_dataset=True, is_training=True, is_task=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, drop_last=False, num_workers=4, )

    images_iter = iter(valid_loader)
    sample = next(images_iter)
    images, label = sample['imgs'].float(), sample['labels'].float()

    imgs = []
    activation_maps = []

    for index in range(args.batch_size):
        imgs.append(to_pil_image(images[index].squeeze(0)))
        images_tensor = images[index]

        if torch.cuda.is_available():
            model = model.cuda()
        if torch.cuda.is_available():
            images_tensor = images_tensor.unsqueeze(0).cuda()
        if args.cam_method is not None:
            cam_extractor = methods.__dict__[args.cam_method](model,
                                                              target_layer=[s.strip() for s in
                                                                            args.target_layer.split("+")] if len(
                                                                  args.target_layer) > 0 else None
                                                              )
        out = model(images_tensor)
        weight, activation, act_maps = cam_extractor(temp_target_class, out)
        activation_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
        activation_maps.append(activation_map)

        weights_list = weight.squeeze(3).squeeze(2).squeeze(0).tolist()
        max_number = heapq.nlargest(int(len(weights_list) * args.proportion), weights_list)
        mask_index_item = [0] * len(weights_list)
        for t in max_number:
            temp = weights_list.index(t)
            mask_index_item[temp] = 1

    return imgs, activation, activation_maps, mask_index_item


def show_cams(name, imgs, activation_maps, save=False, filepath=""):
    columns = int(len(activation_maps) ** 0.5)
    rows = int(len(activation_maps) ** 0.5)
    fig = plt.figure(figsize=(columns, rows))
    #plt.title(name)
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        result = overlay_mask(imgs[i - 1], to_pil_image(activation_maps[i - 1], mode="F"), alpha=0.5)
        plt.imshow(result)
    if save:
        plt.savefig(filepath)
    else:
        plt.show()


def main(args):
    args.saved_model_path = os.path.join(args.saved_model_path, args.run_name)
    pretrained_adversary_network_path = os.path.join(args.saved_model_path, args.run_name + "_final.pth")

    if os.path.exists(pretrained_adversary_network_path):
        index_model = torch.load(pretrained_adversary_network_path)
        cprint.warn("Load Checkpoint" + pretrained_adversary_network_path)

    else:
        cprint.warn("No Checkpoint File Found " + pretrained_adversary_network_path)


    # index all layers
    model_layer_names = []
    for name, m in index_model.named_modules():
        if isinstance(m, nn.Conv2d):
            model_layer_names.append(name)

    # create the basis graph
    start_time = time.time()


    myexplanatoryGraph = explantorygraph()
    for name, m in index_model.named_modules():
        if isinstance(m, nn.Conv2d) and name in model_layer_names[len(model_layer_names) - 6:len(model_layer_names)-1]:
            args.target_layer = name
            my_explanatorylayer = explanatorylayer(name, m.weight.shape[0])

            for temp_target_class in range(args.total_class_number):

                if temp_target_class in args.target_class:
                    if args.tv_model is not None:
                        pretrained_adversary_network_path = os.path.join(args.saved_model_path, args.run_name + "_final.pth")
                        target_model = torch.load(pretrained_adversary_network_path)
                        target_model.eval().cuda()
                    imgs, activation, activation_maps, mask_index_item = evaluate_cam_classs_idx(args, temp_target_class, target_model)
                    #show_cams(None, imgs, activation_maps, True, "cams"+"/"+args.run_name + "_" + str(args.target_layer) + "_withbalance.png")
                    for i in range(len(mask_index_item)):
                        if mask_index_item[i] == 1:
                            my_explanatorylayer.nodes_weights[i] = my_explanatorylayer.nodes_weights[i] + args.total_class_number
                        else:
                            pass
                else:
                    if args.tv_model is not None:
                        pretrained_adversary_network_path = os.path.join(args.saved_model_path, args.run_name + "_final.pth")
                        target_model = torch.load(pretrained_adversary_network_path)
                        target_model.eval().cuda()
                    imgs, activation, activation_maps, mask_index_item = evaluate_cam_classs_idx(args, temp_target_class, target_model)
                    for i in range(len(mask_index_item)):
                        if mask_index_item[i] == 1:
                            my_explanatorylayer.nodes_weights[i] = my_explanatorylayer.nodes_weights[i] - 1
                        else:
                            pass
            myexplanatoryGraph.add_layer(my_explanatorylayer)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"cost: {elapsed_time:.2f}s")


    myexplanatoryGraph.print_graph()
    myexplanatoryGraph.save_target_class(args)
    myexplanatoryGraph.load_target_class(args)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Show CAM in Multi Setting')
    parser.add_argument('--data_path', type=str, default='../../datasets/CelebaK/', help='Root path of data')
    parser.add_argument('--saved_model_path', type=str, default='./checkpoints/original_multi/', help='Dir path to save model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for testing (default: 1)')
    parser.add_argument('--total_class_number', type=int, default=None, help='which image you would like to show')
    parser.add_argument('--tv_model', type=str, default="resnet18", help='which model you would like to use')
    parser.add_argument('--dataset', type=str, default='celeba', help='dataset?')
    parser.add_argument('--target_layer', type=str, default="block6_2.projection", help='which layer you would like to show')
    parser.add_argument('--cam_method', type=str, default="GradCAM", help='which cam method you would like to use')
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--run_name', type=str, default=None)

    #Setting IV
    parser.add_argument('--task_labels', type=list, default=['Smiling','No_Beard','Eyeglasses'], help='Attributes')
    parser.add_argument('--proportion', type=float, default=0.1, help='Learning Rate')
    parser.add_argument('--target_class', type=list, default=[1,2], help='which image you would like to show')

    args = parser.parse_args()
    args.run_name = '_'.join(args.task_labels)
    args.total_class_number = len(args.task_labels)
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    main(args)
