import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp # Transfer Learning Library

# Import your updated dataset
from dataset2 import RetinalDataset

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8 
EPOCHS = 40
LEARNING_RATE = 1e-3

TRAIN_IMG_DIR = "data/Training_Set"
TRAIN_MASK_DIR = "pseudo_masks/train"
VAL_IMG_DIR = "data/Validation_Set"
VAL_MASK_DIR = "pseudo_masks/val"

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# 2. Advanced Hybrid Loss (Focal + Dice)
# ==========================================
# SMP provides highly optimized, built-in versions of these math functions
focal_loss = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
dice_loss = smp.losses.DiceLoss(mode='binary')

def hybrid_focal_dice_loss(preds, targets):
    # Focal handles the thin capillaries, Dice handles the geometric shape
    return focal_loss(preds, targets) + dice_loss(preds, targets)

# ==========================================
# 3. Training Function
# ==========================================
def train_model():
    print(f"🚀 Initializing Advanced U-Net Training on {DEVICE}...")
    
    # Load Data
    train_ds = RetinalDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, is_train=True)
    val_ds = RetinalDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Pre-Trained Encoder (Strategy 4)
    print("📥 Loading Pre-trained EfficientNet-B0 Encoder...")
    model = smp.Unet(
        encoder_name="efficientnet-b0", 
        encoder_weights="imagenet",   # Starts with millions of images worth of "knowledge"
        in_channels=3,                  
        classes=1                      
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device='cuda')
    
    # Safety Net
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    best_val_dice = 0.0
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            
            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = hybrid_focal_dice_loss(preds, masks)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation Phase
        model.eval()
        val_dice_score = 0
        with torch.no_grad():
            for val_imgs, val_masks in val_loader:
                val_imgs = val_imgs.to(DEVICE)
                val_masks = val_masks.to(DEVICE)
                
                val_preds = model(val_imgs)
                
                # SMP DiceLoss outputs a loss (closer to 0 is better). 
                # We subtract from 1 to get the actual positive Score (closer to 1 is better)
                current_dice = 1 - dice_loss(val_preds, val_masks).item()
                val_dice_score += current_dice
                
        val_dice_score /= len(val_loader)
        print(f"🌟 Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {val_dice_score:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        
        scheduler.step(val_dice_score)
        
        if val_dice_score > best_val_dice:
            best_val_dice = val_dice_score
            torch.save(model.state_dict(), "checkpoints/unet_best_advanced.pth")
            print("💾 New Best Advanced Model Saved!")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support() 
    train_model()