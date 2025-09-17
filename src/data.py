from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter

def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 2):
    train_tfms, val_tfms = get_transforms(img_size)
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_tfms)
    class_names = train_ds.classes

    # --- Balanced sampler for training ---
    # count per-class
    counts = Counter([y for _, y in train_ds.samples])
    # weight for each class = 1 / count
    class_weights = {cls_idx: 1.0 / counts[cls_idx] for cls_idx in counts}
    # sample weight for each item
    sample_weights = np.array([class_weights[y] for _, y in train_ds.samples], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, class_names

def get_transforms(img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms

