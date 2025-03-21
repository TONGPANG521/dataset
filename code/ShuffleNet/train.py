import os
import math
import argparse
import pickle
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x0_5
from my_dataset import MyDataSet
from utils import read_data, train_one_epoch, evaluate

train_root = "train_path"  
val_root = "validation_path"  

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    train_images_path, train_images_label = read_data(train_root)
    val_images_path, val_images_label = read_data(val_root)

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = shufflenet_v2_x0_5(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=1.0, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    for epoch in range(args.epochs):
        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        print(f"[epoch {epoch}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accuracy, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)

        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_loss': train_loss_values,
            'train_accuracy': train_accuracy_values,
            'val_loss': val_loss_values,
            'val_accuracy': val_accuracy_values
        }, f)

    tb_writer.close()

def plot_metrics():
    with open('training_metrics.pkl', 'rb') as f:
        data = pickle.load(f)
        train_loss_values = data['train_loss']
        train_accuracy_values = data['train_accuracy']
        val_loss_values = data['val_loss']
        val_accuracy_values = data['val_accuracy']

    epochs = len(train_loss_values)
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 6))

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

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(epochs_range, train_loss_values, label='Training Loss')
    plt.plot(epochs_range, val_loss_values, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(0, epochs + 1)
    plt.ylim(0, max(max(train_loss_values), max(val_loss_values)))

    plt.tight_layout(pad=5.0)
    plt.savefig('training_validation_plots.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--weights', type=str, default='./Shu.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args(args=[])

    main(opt)
    plot_metrics()
