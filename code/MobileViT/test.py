import os
import torch
from PIL import Image
from torchvision import transforms
from model import mobile_vit_xx_small as create_model

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(2)


def preprocess_image(img_path):
    assert os.path.exists(img_path), f"file '{img_path}' does not exist."
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure images are resized to match input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Match training normalization
    ])
    img = preprocess(img).unsqueeze(0)  
    return img


def predict_single_image(img_path, model, device):
    img = preprocess_image(img_path).to(device)
    model.eval() 
    with torch.no_grad():
        result = model(img)
        predict_class = result.argmax(dim=1).item()
    return predict_class


def predict_images_in_folder(folder_path, model, device):
    predictions = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                img_path = os.path.join(root, filename)
                class_index = predict_single_image(img_path, model, device)
                predictions.append(class_index)
    return predictions


num_classes = 4  # Adjust this according to your training setup
weights_path = './weights/latest_model.pth'  
assert os.path.exists(weights_path), f"cannot find {weights_path}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = create_model(num_classes=num_classes).to(device)


pretrained_dict = torch.load(weights_path, map_location=device)
model_dict = model.state_dict()


pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


folder_path = "data_path"  
assert os.path.exists(folder_path), f"file '{folder_path}' does not exist."

all_predictions = predict_images_in_folder(folder_path, model, device)

print(all_predictions)
