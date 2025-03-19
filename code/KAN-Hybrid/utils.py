import os
import json
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def read_data(root: str):
    assert os.path.exists(root), "dataset file: {} does not exist.".format(root)

    crack_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    crack_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(crack_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  
    images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"] 

    for cla in crack_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("Found {} images in the dataset.".format(sum(every_class_num)))

    return images_path, images_label


class CrackDataset(Dataset):
    def __init__(self, images_path, images_label, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = Image.open(self.images_path[index]).convert('RGB')
        label = self.images_label[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def generate_ds(train_root: str,
                val_root: str,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 10,
                cache_data: bool = False):

    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    train_img_path, train_img_label = read_data(train_root)
    val_img_path, val_img_label = read_data(val_root)

    train_transform = transforms.Compose([
        transforms.Resize((train_im_height, train_im_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((val_im_height, val_im_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = CrackDataset(train_img_path, train_img_label, transform=train_transform)
    val_dataset = CrackDataset(val_img_path, val_img_label, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
