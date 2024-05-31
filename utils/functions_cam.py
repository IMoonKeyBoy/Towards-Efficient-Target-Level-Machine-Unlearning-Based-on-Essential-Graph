
import sys
import heapq
import torch
sys.path.append('..')
from pathlib import Path
from torchcam import methods
from tqdm import trange, tqdm
from torchvision import models
import torch.nn.functional as F
#from functions import UnNormalize
from matplotlib import pyplot as plt
from torchcam.utils import overlay_mask
from torchvision import transforms as tfm
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

def generate_one_samples(dataset, list_classes: list):
    sub_dataset = []
    for datapoint in tqdm(dataset):
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:
            list_classes.remove(label_index)
            sub_dataset.append(datapoint)
            break
    return sub_dataset

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(std)[:, None, None]

    def __call__(self, x):
        return (x * self.std) + self.mean
unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



def calculate_importance(args, model,images_tensor) -> list:
    trans = tfm.Compose([
        tfm.Resize((224, 224)),  # here, tfm.Resize((224, 224)), is better than CenterCrop
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    topil = tfm.ToPILImage()
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    toimg = tfm.Compose([
        unnorm,
        topil,
    ])

    # Model selection
    if args.tv_model is not None:
        model = models.__dict__[args.tv_model](weights="DEFAULT").eval()
    if torch.cuda.is_available():
        model = model.cuda()
    if args.cam_method is not None:
        cam_extractor = methods.__dict__[args.cam_method](model,
                                                          target_layer=[s.strip() for s in
                                                                        args.target_layer.split("+")] if len(
                                                              args.target_layer) > 0 else None
                                                          )
    # imgs.append(to_pil_image(unnorm(images[index]).squeeze(0)))
    if torch.cuda.is_available():
        images_tensor = images_tensor.cuda()
    # Forward the image to the model
    out = model(images_tensor.unsqueeze(0))
    # Select the target class
    class_idx = out.squeeze(0).argmax().item()
    # Retrieve the CAM
    weight, activation, act_maps = cam_extractor(class_idx, out)
    # Calculate the most important weight's index
    weights_list = weight.squeeze(3).squeeze(2).squeeze(0).tolist()

    max_number = heapq.nlargest(int(len(weights_list) * args.proportion), weights_list)
    mask_index_item = [0] * len(weights_list)
    for t in max_number:
        temp = weights_list.index(t)
        mask_index_item[temp] = 1
    return mask_index_item

def calculate_importance_with_image(args, model) -> list:


    topil = tfm.ToPILImage()
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    toimg = tfm.Compose([
        unnorm,
        topil,
    ])

    # Get and Preprocess image
    trans = tfm.Compose([
        tfm.Resize((224, 224)),  # here, tfm.Resize((224, 224)), is better than CenterCrop
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    TEST_DATA_PATH = Path(__file__).resolve()
    calculation_mask_dataset = ImageFolder(
        TEST_DATA_PATH.home() / "Data/PycharmWorkspace/datasets/imagenet-mini/train", trans)
    images_iter = iter(DataLoader(calculation_mask_dataset, batch_size=args.calculate_mask_batch_size, shuffle=False))
    for i in range(args.index_number):
        images, label = next(images_iter)

    if args.cam_method is not None:
        cam_extractor = methods.__dict__[args.cam_method](model,
                                                          target_layer=[s.strip() for s in
                                                                        args.target_layer.split("+")] if len(
                                                              args.target_layer) > 0 else None
                                                          )
    # imgs.append(to_pil_image(unnorm(images[index]).squeeze(0)))
    if torch.cuda.is_available():
        images_tensor = images.cuda()

    # Forward the image to the model
    out = model(images_tensor)
    # Select the target class
    class_idx = out.squeeze(0).argmax().item()
    print(class_idx, end="\t")
    # Retrieve the CAM
    weight, activation, act_maps = cam_extractor(class_idx, out)
    # Calculate the most important weight's index
    weights_list = weight.squeeze(3).squeeze(2).squeeze(0).tolist()
    max_number = heapq.nlargest(int(len(weights_list) * args.proportion), weights_list)
    mask_index_item = [0] * len(weights_list)
    for t in max_number:
        temp = weights_list.index(t)
        mask_index_item[temp] = 1
        # mask_index_item.append(temp)
        # weights_list[temp] = -1
    #print(mask_index_item)
    return mask_index_item

def calculate_importance_return_all(args, model,images_tensor) -> list:
    trans = tfm.Compose([
        tfm.Resize((224, 224)),  # here, tfm.Resize((224, 224)), is better than CenterCrop
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    topil = tfm.ToPILImage()
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    toimg = tfm.Compose([
        unnorm,
        topil,
    ])

    # Model selection
    if args.tv_model is not None:
        model = models.__dict__[args.tv_model](weights="DEFAULT").eval()
    if torch.cuda.is_available():
        model = model.cuda()
    if args.cam_method is not None:
        cam_extractor = methods.__dict__[args.cam_method](model,
                                                          target_layer=[s.strip() for s in
                                                                        args.target_layer.split("+")] if len(
                                                              args.target_layer) > 0 else None
                                                          )
    # imgs.append(to_pil_image(unnorm(images[index]).squeeze(0)))
    if torch.cuda.is_available():
        images_tensor = images_tensor.cuda()

    # Forward the image to the model
    out = model(images_tensor.unsqueeze(0))
    # Select the target class
    class_idx = out.squeeze(0).argmax().item()
    # Retrieve the CAM
    weight, activation, act_maps = cam_extractor(class_idx, out)
    # Calculate the most important weight's index
    weights_list = weight.squeeze(3).squeeze(2).squeeze(0).tolist()
    max_number = heapq.nlargest(int(len(weights_list) * args.proportion), weights_list)
    mask_index_item = [0] * len(weights_list)
    for t in max_number:
        temp = weights_list.index(t)
        mask_index_item[temp] = 1
        # mask_index_item.append(temp)
        # weights_list[temp] = -1
    print(mask_index_item)
    return weight, activation, act_maps, mask_index_item

def evaluate_cam(args, model):

    # Get and Preprocess image
    trans = tfm.Compose([
        tfm.Resize((224, 224)),  # here, tfm.Resize((224, 224)), is better than CenterCrop
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    TEST_DATA_PATH = Path(__file__).resolve()
    calculation_mask_dataset = ImageFolder(
        TEST_DATA_PATH.home() / "Data/PycharmWorkspace/datasets/imagenet-mini/train", trans)
    images_iter = iter(DataLoader(calculation_mask_dataset, batch_size=args.calculate_mask_batch_size, shuffle=False))
    for i in range(args.index_number):
        images, label = next(images_iter)

    imgs = []
    activation_maps = []

    for index in range(args.calculate_mask_batch_size):
        imgs.append(to_pil_image(unnorm(images[index]).squeeze(0)))
        images_tensor = images[index]
        if torch.cuda.is_available():
            images_tensor = images_tensor.cuda()

        if args.cam_method is not None:
            cam_extractor = methods.__dict__[args.cam_method](model,
                                                              target_layer=[s.strip() for s in
                                                                            args.target_layer.split("+")] if len(
                                                                  args.target_layer) > 0 else None
                                                              )
        # Forward the image to the model
        out = model(images_tensor.unsqueeze(0))
        # Select the target class
        class_idx = out.squeeze(0).argmax().item()
        # Retrieve the CAM
        print(class_idx)
        weight, activation, act_maps = cam_extractor(class_idx, out)
        # Fuse the CAMs if there are several
        activation_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
        activation_maps.append(activation_map)

    return imgs, weight, activation, activation_maps

def evaluate_cam_classs_idx(args, model, classidx):

    # Get and Preprocess image
    trans = tfm.Compose([
        tfm.Resize((224, 224)),  # here, tfm.Resize((224, 224)), is better than CenterCrop
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    TEST_DATA_PATH = Path(__file__).resolve()
    calculation_mask_dataset = ImageFolder(
        TEST_DATA_PATH.home() / "Data/PycharmWorkspace/datasets/imagenet-mini/train", trans)
    #dataset_one_class = generate_one_samples(calculation_mask_dataset,[243])
    images_iter = iter(DataLoader(calculation_mask_dataset, batch_size=args.calculate_mask_batch_size, shuffle=False))
    for i in range(args.index_number):
        images, label = next(images_iter)

    imgs = []
    activation_maps = []

    for index in range(args.calculate_mask_batch_size):
        imgs.append(to_pil_image(unnorm(images[index]).squeeze(0)))
        images_tensor = images[index]
        if torch.cuda.is_available():
            images_tensor = images_tensor.unsqueeze(0).cuda()
        if args.cam_method is not None:
            cam_extractor = methods.__dict__[args.cam_method](model,
                                                              target_layer=[s.strip() for s in
                                                                            args.target_layer.split("+")] if len(
                                                                  args.target_layer) > 0 else None
                                                              )
        # Forward the image to the model
        print(images_tensor.shape)
        sys.exit(0)
        out = model(images_tensor)
        #print(out.topk(10))
        # Select the target class
        #class_idx = out.squeeze(0).argmax().item()
        # Retrieve the CAM
        weight, activation, act_maps = cam_extractor(classidx, out)
        # Fuse the CAMs if there are several
        activation_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
        activation_maps.append(activation_map)

    return imgs, weight, activation, activation_maps

def show_important_weighted_feature_original_size(weight, activation, index_list,save=False,filepath = ""):
    weighted_featuremaps = weight * activation
    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int((weighted_featuremaps[0].shape[0] ** 0.5))
    print(columns)
    rows = int((weighted_featuremaps[0].shape[0] ** 0.5))
    fig = plt.figure(figsize=(columns, rows))

    for i in trange(1, columns * rows + 1):
        if index_list[i - 1] == 1:
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(to_pil_image(weighted_featuremaps[0][i - 1].cpu()))
    if save:
        plt.savefig(filepath)
    else:
        plt.show()

def show_important_weighted_feature_resize(weight, activation, index_list):
    weighted_featuremaps = weight * activation
    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int(sum(i == 1 for i in index_list) ** 0.5)
    rows = int(sum(i == 1 for i in index_list) ** 0.5)
    fig = plt.figure(figsize=(columns, rows))
    image_index  = 1
    for i in trange(len(index_list)):
        if index_list[i] == 1:
            fig.add_subplot(rows, columns,image_index)
            image_index = image_index + 1
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(to_pil_image(weighted_featuremaps[0][i].cpu()),cmap="gray")
    plt.show()

def show_remaining_weighted_feature_original_size(weight, activation, index_list,save=False,filepath = ""):
    weighted_featuremaps = weight * activation
    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int((weighted_featuremaps[0].shape[0] ** 0.5))
    rows = int((weighted_featuremaps[0].shape[0] ** 0.5))
    fig = plt.figure(figsize=(columns, rows))
    for i in trange(1, columns * rows + 1):
        if index_list[i - 1] != 1:
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(to_pil_image(weighted_featuremaps[0][i - 1].cpu()))
    if save:
        plt.savefig(filepath)
    else:
        plt.show()

def show_remaining_weighted_feature_resize(weight, activation, index_list):
    weighted_featuremaps = weight * activation
    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int(sum(i == 0 for i in index_list) ** 0.5)
    rows = int(sum(i == 0 for i in index_list) ** 0.5)
    fig = plt.figure(figsize=(columns, rows))
    image_index = 1
    for i in trange(len(index_list)):
        if index_list[i] != 1:
            fig.add_subplot(rows, columns,image_index)
            image_index = image_index + 1
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(to_pil_image(weighted_featuremaps[0][i].cpu()),cmap="gray")
    plt.show()

def show_weighted_feature_maps(weight, activation):
    weighted_featuremaps = weight * activation

    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int((weighted_featuremaps[0].shape[0] ** 0.5))
    rows = int((weighted_featuremaps[0].shape[0] ** 0.5))

    fig = plt.figure(figsize=(columns, rows))
    for i in trange(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        # weighted_featuremaps[weighted_featuremaps < 0.005] = 0
        # print(np.count_nonzero(weighted_featuremaps[0][i - 1].cpu().numpy()))
        # print(weighted_featuremaps[0][i-1])
        plt.imshow(to_pil_image(weighted_featuremaps[0][i - 1].cpu()))
    plt.show()

def show_feature_maps(activation):
    weighted_featuremaps = F.relu(activation, inplace=True)
    columns = int((weighted_featuremaps[0].shape[0] ** 0.5))
    rows = int((weighted_featuremaps[0].shape[0] ** 0.5))
    fig = plt.figure(figsize=(columns, rows))
    for i in trange(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        # weighted_featuremaps[weighted_featuremaps < 0.005] = 0
        # print(np.count_nonzero(weighted_featuremaps[0][i - 1].cpu().numpy()))
        # print(weighted_featuremaps[0][i-1])
        plt.imshow(to_pil_image(weighted_featuremaps[0][i - 1].cpu()))
    plt.show()

def show_weighted_feature_maps_norm(weight, activation):
    weighted_featuremaps = weight * activation
    weighted_featuremaps = F.relu(weighted_featuremaps, inplace=True)
    columns = int((weighted_featuremaps[0].shape[0] ** 0.5))
    rows = int((weighted_featuremaps[0].shape[0] ** 0.5))
    fig = plt.figure(figsize=(columns, rows))
    for i in trange(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        # weighted_featuremaps[weighted_featuremaps < 0.005] = 0
        # print(np.count_nonzero(weighted_featuremaps[0][i - 1].cpu().numpy()))
        # print(weighted_featuremaps[0][i-1])
        plt.imshow(to_pil_image(normalize(weighted_featuremaps[0][i - 1]).cpu()),cmap="gray")
    plt.show()

def show_cams(imgs,activation_maps,save=False,filepath = ""):
    # Plot CAM
    columns = int(len(activation_maps)** 0.5)
    rows = int(len(activation_maps)** 0.5)
    fig = plt.figure(figsize=(columns, rows))
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

def show_activations(activation_maps):

    columns = int(len(activation_maps) ** 0.5)
    rows = int(len(activation_maps) ** 0.5)
    fig = plt.figure(figsize=(columns, rows))
    fig.tight_layout()
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(to_pil_image(activation_maps[0].cpu()))
    plt.show()

def show_imgs(images):
    imgs = []
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for i in range(len(images)):
        imgs.append(to_pil_image(unnorm(images[i].clone()).squeeze(0)))
    # Plot the raw image
    columns = int(len(imgs)**0.5)
    rows = int(len(imgs)**0.5)
    fig = plt.figure(figsize=(columns, rows))
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(imgs[i - 1],cmap="gray")
    plt.show()

def show_only_cams(cams):
    columns = int(len(cams))
    rows = 1
    fig = plt.figure(figsize=(columns, rows))
    fig.tight_layout()
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(to_pil_image(cams[i - 1].cpu()))
    plt.show()