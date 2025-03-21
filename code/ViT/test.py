import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from vit_model import vit_base_patch16_224 as create_model

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def preprocess_image(img_path):
    assert os.path.exists(img_path), f"file '{img_path}' does not exist."
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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

num_classes = 4
weights_path = './save_weights/model.pth'
assert os.path.exists(weights_path), f"cannot {weights_path}"

model = create_model(num_classes=num_classes, has_logits=False)
model.head = torch.nn.Linear(768, num_classes)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()  

folder_path = "data_path"  
assert os.path.exists(folder_path), f"folder '{folder_path}' does not exist."

all_predictions = predict_images_in_folder(folder_path, model, device)

print(all_predictions)
