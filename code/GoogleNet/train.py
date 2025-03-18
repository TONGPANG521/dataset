import os
import sys
import json
import pickle
import torchvision
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 输入数据集路径
    train_data_path = 'Replace it with your train data path'  
    val_data_path = 'Replace it with your validation data path'   

    assert os.path.exists(train_data_path), "{} path does not exist.".format(train_data_path)
    assert os.path.exists(val_data_path), "{} path does not exist.".format(val_data_path)

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transform["train"])
    train_num = len(train_dataset)

    # Create a category index mapping and save it as a JSON file.
    crack_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in crack_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 50
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    
    net = torchvision.models.googlenet(num_classes=4)
    model_dict = net.state_dict()
    # Pretrained weights download link: https://download.pytorch.org/models/googlenet-1378be20.pth
    pretrain_model = torch.load("googlenet-1378be20.pth")
    del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
                "aux2.fc2.weight", "aux2.fc2.bias",
                "fc.weight", "fc.bias"]
    pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    model_dict.update(pretrain_dict)
    net.load_state_dict(model_dict)
    
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)


    epochs = 100
    best_acc = 0.0
    save_path = './googleNet.pth'
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
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.to(device)).sum().item()

        train_loss = running_loss / train_steps
        train_accuracy = correct_train / total_train
        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)

        
        net.eval()
        acc = 0.0  
        running_val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  
                loss = loss_function(outputs, val_labels.to(device))
                running_val_loss += loss.item()

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_loss = running_val_loss / len(validate_loader)
        val_accuracy = acc / val_num
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    # Save training and validation metrics.
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({
            'train_loss': train_loss_values,
            'train_accuracy': train_accuracy_values,
            'val_loss': val_loss_values,
            'val_accuracy': val_accuracy_values
        }, f)

    print('Finished Training')

    # Plot loss and accuracy graphs.
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

    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(epochs_range, train_accuracy_values, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy_values, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, epochs + 1)  

    plt.tight_layout(pad=5.0)  
    plt.savefig('training_validation_plots.png')  
    plt.show()  

if __name__ == '__main__':
    main()
