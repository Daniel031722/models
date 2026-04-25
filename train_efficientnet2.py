import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  
EPOCHS = 30
LEARNING_RATE = 1e-4 

# Paths to your perfectly generated masks
IMG_DIRS = {
    "train": "fusion_data/train",
    "val": "fusion_data/val"
}

# Pointing exactly to your RFMiD label files
CSV_PATHS = {
    "train": "a. RFMiD_Training_Labels.csv", 
    "val": "b. RFMiD_Validation_Labels.csv"
}

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# ⭐ ADVANCED LOSS: Binary Focal Loss 
# (Extracted from your old code)
# ==========================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==========================================
# 2. Dataset Logic
# ==========================================
class RiskClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Grab the raw ID from the CSV
        raw_name = str(self.data.iloc[idx, 0])
        base_name = os.path.splitext(raw_name)[0] # Strip off any existing extension
        
        # Check for both .jpg and .png formats automatically
        img_path_jpg = os.path.join(self.img_dir, base_name + '.jpg')
        img_path_png = os.path.join(self.img_dir, base_name + '.png')
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            img_path = img_path_png
            
        label = float(self.data.iloc[idx, 1])
        
        # Read the image
        image = cv2.imread(img_path)
        
        # Safety check to prevent that cvtColor crash
        if image is None:
            raise FileNotFoundError(f"Could not find image for {base_name} in {self.img_dir}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. Augmentations
# ==========================================
train_transform = A.Compose([
    A.Resize(300, 300),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(300, 300),
    ToTensorV2()
])

# ==========================================
# 4. Training Engine
# ==========================================
def train_classification_model():
    print(f"🚀 Initializing Ultimate EfficientNet-B3 Classifier on {DEVICE}...")
    
    # --- DYNAMIC FOCAL LOSS ALPHA CALCULATION ---
    print("⚖️ Analyzing dataset for Focal Loss Alpha...")
    train_df = pd.read_csv(CSV_PATHS["train"])
    num_zeros = len(train_df[train_df.iloc[:, 1] == 0.0])
    num_ones = len(train_df[train_df.iloc[:, 1] == 1.0])
    
    # In Focal loss, alpha is the weight for the positive class (1). 
    # To prioritize the rare 0s, alpha should be the ratio of 0s to the total.
    alpha_val = num_zeros / (num_zeros + num_ones)
    print(f"   Healthy (0): {num_zeros} | At Risk (1): {num_ones} | Alpha set to: {alpha_val:.4f}")
    
    criterion = BinaryFocalLoss(alpha=alpha_val, gamma=2.0).to(DEVICE)
    # ----------------------------------------

    # 1. Load Data
    train_ds = RiskClassificationDataset(CSV_PATHS["train"], IMG_DIRS["train"], transform=train_transform)
    val_ds = RiskClassificationDataset(CSV_PATHS["val"], IMG_DIRS["val"], transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Build the Model
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    
    # ⭐ UPGRADE: Injecting the Dropout Layer from old code
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, 1)
    )
    model = model.to(DEVICE)
    
    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device='cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = criterion(preds, labels)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            predicted_classes = (torch.sigmoid(preds) > 0.5).float()
            correct_train += (predicted_classes == labels).sum().item()
            total_train += labels.size(0)
            loop.set_postfix(loss=loss.item())
            
        train_acc = correct_train / total_train
        
        # 5. Validation Phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs, val_labels = val_imgs.to(DEVICE), val_labels.to(DEVICE).unsqueeze(1)
                
                val_preds = model(val_imgs)
                batch_loss = criterion(val_preds, val_labels)
                val_loss += batch_loss.item()
                
                val_predicted_classes = (torch.sigmoid(val_preds) > 0.5).float()
                correct_val += (val_predicted_classes == val_labels).sum().item()
                total_val += val_labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        
        print(f"🌟 Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/efficientnet_b3_best.pth")
            print("💾 New Best Ultimate Classifier Saved!")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support() 
    train_classification_model()