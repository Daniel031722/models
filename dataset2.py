import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        if is_train:
            self.transform = A.Compose([
                # 1. Patch-Based Training: RandomCrop preserves tiny capillaries
                A.Resize(384, 384),
                
                # 2. Anatomical Augmentations: Mimics biological stretching
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(p=0.3),
                
                # Standard spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384, 384), # Keep validation at full view to accurately compare with previous runs
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32) 

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)

        return image, mask