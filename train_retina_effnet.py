import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# Paths
train_dir = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\Training'
train_label_csv = r'C:\Users\Server\Documents\retinal-diseases\database\Training_Set\Training_Set\RFMiD_Training_Labels.csv'

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
IMG_SIZE = 224

# Load CSV and split
df = pd.read_csv(train_label_csv)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class RetinaDataset(Dataset):
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
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = RetinaDataset(train_df, train_dir, transform)
val_dataset = RetinaDataset(val_df, train_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
num_classes = len(df.columns) - 1
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'efficientnet_retina.pth')
