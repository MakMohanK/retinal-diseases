import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATH CONFIGURATION
# ===============================
input_folder = r'/home/pi/Desktop/keegan_project/input'
model_path = r'/home/pi/Desktop/keegan_project/efficientnet_retina.pth'
test_label_csv = r'/home/pi/Desktop/keegan_project/RFMiD_Training_Labels.csv'

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL LOADING
# ===============================
num_classes = len(pd.read_csv(test_label_csv).columns) - 1
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return probs

# ===============================
# RUN PREDICTIONS ON ALL IMAGES
# ===============================
disease_labels = pd.read_csv(test_label_csv).columns[1:]
results = []

image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    probs = predict_image(img_path)
    predictions = {label: float(prob)
                   for label, prob in zip(disease_labels, probs)}
    results.append({"image": img_name, **predictions})

    # ===============================
    # VISUALIZE RESULTS FOR THIS IMAGE
    # ===============================
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

    top_detections = {k: v for k, v in sorted_preds.items() if v > 0.4}

    # Plot two subplots horizontally
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Predictions for {img_name}", fontsize=14, fontweight='bold')

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

# ===============================
# SAVE ALL PREDICTIONS TO CSV
# ===============================
output_csv = os.path.join(input_folder, "retina_predictions.csv")
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n✅ Predictions saved to: {output_csv}")
