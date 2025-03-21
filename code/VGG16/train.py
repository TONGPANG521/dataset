import os
import sys
import json
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import vgg  

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_path = 'train_path'
    val_path = 'validation_path'

    assert os.path.exists(train_path), f"Training path {train_path} does not exist."
    assert os.path.exists(val_path), f"Validation path {val_path} does not exist."

    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    crack_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in crack_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 20
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=4, init_weights=False)

    pretrained_weights_path = '16.pth'
    if os.path.exists(pretrained_weights_path):
        print(f"Loading pretrained weights from {pretrained_weights_path}")

        state_dict = torch.load(pretrained_weights_path)

        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}

        net.load_state_dict(state_dict, strict=False)
    else:
        print(f"Pretrained weights file {pretrained_weights_path} not found. Proceeding with randomly initialized weights.")

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    
    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)

    train_loss_values = []
    val_loss_values = []
    train_accuracy_values = []
    val_accuracy_values = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_accuracy = correct_train / total_train
        train_loss_values.append(running_loss / len(train_loader))
        train_accuracy_values.append(train_accuracy)

        
        net.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)  
                outputs = net(val_images)
                val_loss = loss_function(outputs, val_labels)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()

        val_accuracy = correct_val / total_val
        val_loss_values.append(running_val_loss / len(val_loader))
        val_accuracy_values.append(val_accuracy)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss_values[-1], val_loss_values[-1], train_accuracy, val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

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
    plt.ylim(0, max(max(train_loss_values), max(val_loss_values)) * 1.1)

    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(epochs_range, train_accuracy_values, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, epochs + 1)
    plt.ylim(0, 1.1)

    plt.tight_layout(pad=5.0)
    plt.savefig('training_validation_plots.png')
    plt.show()

if __name__ == '__main__':
    main()
