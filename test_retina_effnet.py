import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

# Paths
test_img_path = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\Training\36.png'
model_path = r'C:\Users\Server\Documents\retinal-diseases\efficientnet_retina.pth'
test_label_csv = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\RFMiD_Training_Labels.csv'

IMG_SIZE = 224

# Device (force CPU if no GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition and load
num_classes = len(pd.read_csv(test_label_csv).columns) - 1
model = efficientnet_b0(weights=None)  # use new API instead of pretrained=True
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
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
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return probs

# Run prediction
disease_labels = pd.read_csv(test_label_csv).columns[1:]
probs = predict_image(test_img_path)
predictions = {label: float(prob) for label, prob in zip(disease_labels, probs)}
print(predictions)

import matplotlib.pyplot as plt

# Convert predictions into sorted dictionary (highest first)
sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

# Colors for risk levels
colors = []
for prob in sorted_preds.values():
    if prob > 0.8:
        colors.append("red")
    elif prob >= 0.4:
        colors.append("orange")
    else:
        colors.append("green")

# Filter top detected diseases (confidence > 0.4)
top_detections = {k: v for k, v in sorted_preds.items() if v > 0.4}

# Plot two subplots horizontally
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Left chart: All diseases with color-coded risk ---
axes[0].bar(list(sorted_preds.keys()), list(sorted_preds.values()), color=colors)
axes[0].set_ylabel("Predicted Probability (Confidence)")
axes[0].set_title("Disease Risks (Color-coded)")
axes[0].tick_params(axis='x', rotation=90)

# --- Right chart: Top detections (>0.4) ---
if top_detections:
    axes[1].bar(list(top_detections.keys()), list(top_detections.values()), color="skyblue")
    axes[1].set_ylabel("Predicted Probability (Confidence)")
    axes[1].set_title("Top Detected Diseases (>0.4)")
    axes[1].tick_params(axis='x', rotation=90)
else:
    axes[1].text(0.5, 0.5, "No diseases > 0.4", 
                 ha="center", va="center", fontsize=12, color="gray")
    axes[1].set_axis_off()

plt.tight_layout()
plt.show()


