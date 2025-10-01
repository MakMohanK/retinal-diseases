import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

# Paths
test_img_path = '/kaggle/input/retinal-disease-classification/Test_Set/Test_Set/Test/123.jpg'  # example image
model_path = 'efficientnet_retina.pth'
test_label_csv = '/kaggle/input/retinal-disease-classification/Test_Set/Test_Set/RFMiD_Testing_Labels.csv'

IMG_SIZE = 224

# Model definition and load
num_classes = len(pd.read_csv(test_label_csv).columns) - 1
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Predict function
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return probs

# Run prediction
disease_labels = pd.read_csv(test_label_csv).columns[1:]
probs = predict_image(test_img_path)
predictions = {label: float(prob) for label, prob in zip(disease_labels, probs)}
print(predictions)
