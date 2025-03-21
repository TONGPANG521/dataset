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

from model import resnet50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }


    data_root = "data_path" 
    train_dir = os.path.join(data_root, "train")  
    val_dir = os.path.join(data_root, "validation")  

    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)
    assert os.path.exists(val_dir), "{} path does not exist.".format(val_dir)


    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    train_num = len(train_dataset)

    validate_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
    val_num = len(validate_dataset)

    # Save class index to a JSON file.
    crack_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in crack_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    batch_size = 20
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))


    net = resnet50()
    model_weight_path = "./R50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))


    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, len(train_dataset.classes))  
    net.to(device)


    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './resNet50.pth'
    train_steps = len(train_loader)


    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train += labels.size(0)
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)


        train_loss = running_loss / train_steps
        train_accuracy = correct_train / total_train
        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)


        net.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                running_val_loss += loss.item()
                total_val += val_labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)


        val_loss = running_val_loss / len(validate_loader)
        val_accuracy = correct_val / total_val
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)


        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_loss': train_loss_values,
            'train_accuracy': train_accuracy_values,
            'val_loss': val_loss_values,
            'val_accuracy': val_accuracy_values
        }, f)

    print('Finished Training')
    

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
    plt.ylim(0, max(max(train_loss_values), max(val_loss_values)) + 0.1)

    

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

if __name__ == '__main__':
    main()
