# ==============================================================
# 🔍 GRAD-CAM FOR EFFICIENTNET RETINA MODEL (IMAGE FILE INPUT)
# ==============================================================

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
IMG_SIZE = 224
MODEL_PATH = "efficientnet_retina.pth"  # your saved model
IMAGE_PATH = r"C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\Training\7.png"  # test image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------
# 1️⃣ MODEL SETUP
# --------------------------------------------------------------
num_classes = 46  # ✅ update this based on your dataset
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Modify classifier for your output
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# --------------------------------------------------------------
# 2️⃣ GRAD-CAM IMPLEMENTATION
# --------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_score = output[0, class_idx]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# --------------------------------------------------------------
# 3️⃣ LOAD AND PREPROCESS IMAGE
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
image_np = np.array(image) / 255.0

# --------------------------------------------------------------
# 4️⃣ GENERATE GRAD-CAM
# --------------------------------------------------------------
target_layer = model.features[-1][0]  # ✅ works for EfficientNet
gradcam = GradCAM(model, target_layer)

# Get prediction
with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
    class_idx = torch.argmax(output, dim=1).item()

# Generate CAM
cam = gradcam.generate(input_tensor, class_idx)

# --------------------------------------------------------------
# 5️⃣ IMPROVED HEATMAP OVERLAY (PRESERVE DIMENSIONS)
# --------------------------------------------------------------
# Generate color heatmap from CAM
cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))  # upscale to original image size
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255.0

# Blend heatmap with original image (in RGB)
overlay = 0.5 * heatmap + 0.5 * image_np[..., ::-1]  # BGR to RGB

# Clip values
overlay = np.clip(overlay, 0, 1)

# --------------------------------------------------------------
# 6️⃣ DISPLAY
# --------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM Overlay")
plt.axis("off")
plt.show()

# Optionally save result
output_path = IMAGE_PATH.replace(".png", "_gradcam.jpg")
cv2.imwrite(output_path, np.uint8(255 * overlay[..., ::-1]))  # convert back to BGR for saving
print(f"✅ Grad-CAM saved at: {output_path}")
