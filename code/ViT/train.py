import os
import re
import sys
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224 as create_model
from utils import generate_ds

def main():
    train_root = "train_path" 
    val_root = "validation_path"    

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    batch_size = 50
    epochs = 100
    num_classes = 4
    freeze_layers = True
    initial_lr = 0.001
    weight_decay = 1e-5

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))
    val_writer = SummaryWriter(os.path.join(log_dir, "val"))

    train_loader, val_loader = generate_ds(train_root, val_root, batch_size=batch_size)

    model = create_model(num_classes=num_classes, has_logits=False)

    pre_weights_path = 'vit_base_patch16_224_in21k.pth'
    assert os.path.exists(pre_weights_path), f"Cannot find {pre_weights_path}"

    pretrained_weights = torch.load(pre_weights_path)

    if 'head.weight' in pretrained_weights:
        del pretrained_weights['head.weight']
    if 'head.bias' in pretrained_weights:
        del pretrained_weights['head.bias']

    model.load_state_dict(pretrained_weights, strict=False)

    model.head = nn.Linear(768, num_classes)  

    if freeze_layers:
        for name, param in model.named_parameters():
            if "pre_logits" not in name and "head" not in name:
                param.requires_grad = False
            else:
                print(f"training {name}")

    model = model.cuda()
    model.train()

    def scheduler(epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  
        new_lr = rate * initial_lr

        train_writer.add_scalar('learning rate', new_lr, epoch)

        return new_lr

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)

    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []

    best_val_acc = 0.
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_description(f"train epoch[{epoch + 1}/{epochs}] loss:{train_loss / total_train:.3f}, acc:{train_correct / total_train:.3f}")

        optimizer.param_groups[0]['lr'] = scheduler(epoch)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        val_bar = tqdm(val_loader, file=sys.stdout)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_bar.set_description(f"valid epoch[{epoch + 1}/{epochs}] loss:{val_loss / total_val:.3f}, acc:{val_correct / total_val:.3f}")

        train_loss_values.append(train_loss / total_train)
        train_accuracy_values.append(train_correct / total_train)

        val_loss_values.append(val_loss / total_val)
        val_accuracy_values.append(val_correct / total_val)

        if val_correct / total_val > best_val_acc:
            best_val_acc = val_correct / total_val
            save_name = "./save_weights/model.pth"
            torch.save(model.state_dict(), save_name)

    import pickle         
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_loss': train_loss_values,
            'train_accuracy': train_accuracy_values,
            'val_loss': val_loss_values,
            'val_accuracy': val_accuracy_values
        }, f)
        
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(epochs_range, train_loss_values, label='Training Loss')
    plt.plot(epochs_range, val_loss_values, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(0, epochs + 1) 
    plt.ylim(0, 0.1)  

    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(epochs_range, train_accuracy_values, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, epochs + 1)  
    plt.ylim(0, 1)  

    plt.tight_layout(pad=5.0)  
    plt.savefig('training_validation_plots.png') 
    plt.show() 


    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    main()
