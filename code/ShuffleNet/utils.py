import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def read_data(root: str):
    assert os.path.exists(root), "data file: {} does not exist".format(root)


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


def train_one_epoch(model, optimizer, train_loader, val_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = 0.0
    val_loss = 0.0
    train_corrects = 0
    val_corrects = 0
    train_total = len(train_loader.dataset)
    val_total = len(val_loader.dataset)

    # Training
    train_loader = tqdm(train_loader, file=sys.stdout, desc=f"[epoch {epoch}] Training")
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = loss_function(pred, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = torch.max(pred, dim=1)[1]
        train_corrects += torch.sum(preds == labels).item()

        train_loader.set_postfix(loss=train_loss / ((step + 1) * images.size(0)),
                                 accuracy=train_corrects / (step + 1))

    train_loss /= train_total
    train_accuracy = train_corrects / train_total

    # Validation
    val_loss, val_accuracy = evaluate(model, val_loader, device)

    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_loss, train_accuracy, val_loss, val_accuracy

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    corrects = 0
    total = len(data_loader.dataset)

    data_loader = tqdm(data_loader, file=sys.stdout, desc="Evaluating")
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_function(pred, labels)
        total_loss += loss.item() * images.size(0)
        preds = torch.max(pred, dim=1)[1]
        corrects += torch.sum(preds == labels).item()

    average_loss = total_loss / total
    accuracy = corrects / total

    return average_loss, accuracy
