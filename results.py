import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import multilabel_confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
train_dir = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\Training'
train_label_csv = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\RFMiD_Training_Labels.csv'
MODEL_PATH = "efficientnet_retina.pth"
RESULTS_DIR = "results_img"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Dataset
# ----------------------------
class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.image_col = 'ID'
        self.label_cols = self.df.columns[1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.loc[idx, self.image_col]) + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(self.df.loc[idx, self.label_cols].values.astype('float32'))
        return image, labels

# Transform
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ----------------------------
# Load CSV and dataset (Validation set only for evaluation)
# ----------------------------
from sklearn.model_selection import train_test_split
df = pd.read_csv(train_label_csv)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

val_dataset = RetinaDataset(val_df, train_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = list(df.columns[1:])
num_classes = len(class_names)

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Evaluate Model
# ----------------------------
all_labels = []
all_preds = []
criterion = nn.BCEWithLogitsLoss()
val_loss = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
avg_val_loss = val_loss / len(val_loader)

print("Validation Loss:", avg_val_loss)
print(classification_report(all_labels, all_preds, target_names=class_names))

# ----------------------------
# Confusion Matrices
# ----------------------------
cms = multilabel_confusion_matrix(all_labels, all_preds)
for i, cm in enumerate(cms):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {class_names[i]}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{RESULTS_DIR}/conf_matrix_{class_names[i]}.png")
    plt.close()

# ----------------------------
# Grad-CAM (Feature Importance Example)
# ----------------------------
def generate_gradcam(model, image_tensor, target_class=0, layer_name="features.6.3"):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    layer = dict([*model.named_modules()])[layer_name]
    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_backward_hook(backward_hook)

    output = model(image_tensor.unsqueeze(0))
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    grads = gradients[0].mean(dim=[2,3], keepdim=True)
    cam = (grads * activations[0]).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = to_pil_image(image_tensor.cpu())
    heatmap_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    fh.remove()
    bh.remove()
    return heatmap_img

# Pick one sample from validation set
sample_img, _ = val_dataset[0]
heatmap_img = generate_gradcam(model, sample_img.to(device), target_class=0)
plt.imsave(f"{RESULTS_DIR}/gradcam_example.png", heatmap_img)
