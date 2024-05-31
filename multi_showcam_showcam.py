import argparse
import random

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


def evaluate_cam_classs_idx(args, model, classidx):
    valid_transforms = get_transforms(args.img_size, mode='valid')
    valid_dataset = CelebASingleClassifiation(args.data_path, task_labels=[args.task_labels[args.target_class]], transforms=valid_transforms, is_split_notask_and_task_dataset=True, is_training=False, is_task=True)
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
            images_tensor = images_tensor.unsqueeze(0).cuda()
        if args.cam_method is not None:
            cam_extractor = methods.__dict__[args.cam_method](model,
                                                              target_layer=[s.strip() for s in
                                                                            args.target_layer.split("+")] if len(
                                                                  args.target_layer) > 0 else None
                                                              )
        out = model(images_tensor)
        weight, activation, act_maps = cam_extractor(classidx, out)
        activation_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
        activation_maps.append(activation_map)

    return imgs,weight, activation, activation_maps


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

    imgs, weight, activation, activation_maps = evaluate_cam_classs_idx(args, index_model, args.target_class)
    show_cams(args.run_name + "_" + str(args.target_class), imgs, activation_maps, False, args.cam_method + "_original.png")


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

    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for testing (default: 1)')

    parser.add_argument('--target_class', type=int, default=1, help='which image you would like to show')
    parser.add_argument('--total_class_number', type=int, default=None, help='which image you would like to show')

    parser.add_argument('--index_number', type=int, default=20, help='which image you would like to show')

    parser.add_argument('--target_layer', type=str, default="block6_2.projection", help='which layer you would like to show')
    parser.add_argument('--cam_method', type=str, default="GradCAM", help='which cam method you would like to use')
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--task_labels', type=list,  default=['Mouth_Slightly_Open','Pointy_Nose'], help='Attributes')

    args = parser.parse_args()
    args.run_name = '_'.join(args.task_labels)
    args.total_class_number = len(args.task_labels)
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    main(args)
