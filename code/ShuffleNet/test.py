import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import shufflenet_v2_x0_5

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
    preprocess = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    image_files = sorted([os.path.join(root, file) 
                          for root, _, files in os.walk(folder_path) 
                          for file in files 
                          if file.endswith(".jpg") or file.endswith(".png")])
    
    predictions = []
    for img_path in image_files:
        class_index = predict_single_image(img_path, model, device)
        predictions.append(class_index)
    
    return predictions

num_classes = 4
weights_path = './weights/model-99.pth'  
assert os.path.exists(weights_path), f"cannot find {weights_path}"

model = shufflenet_v2_x0_5(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval() 

folder_path = "data_path"  
assert os.path.exists(folder_path), f"folder '{folder_path}' does not exist."

all_predictions = predict_images_in_folder(folder_path, model, device)

print(all_predictions)
