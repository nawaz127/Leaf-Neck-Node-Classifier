
import os, argparse, json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from .model import build_model
from .utils import load_config

def load_classes(path: str):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_test_loader(data_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 2):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(f"{data_dir}/test", transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, loader

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    ap = argparse.ArgumentParser(description='Evaluate on test set and save confusion matrix + classification report')
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--weights', type=str, default='models/best.pt')
    ap.add_argument('--classes', type=str, default='models/classes.txt')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--normalize_cm', action='store_true', help='(kept for compatibility; we save both versions)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg['data_dir']
    img_size = cfg.get('img_size', 224)
    model_name = cfg.get('model_name', 'resnet18')

    classes = load_classes(args.classes)

    ds, loader = get_test_loader(data_dir, img_size=img_size, batch_size=args.batch_size)
    assert ds.classes == classes, f"Class mismatch. Test set classes {ds.classes} != {classes}"

    model, _ = build_model(model_name, num_classes=len(classes), pretrained=False)
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # add near other imports
    import csv

    # ...after computing y_true, y_pred, report, cm...
    probs_all = []  # add this above inference loop

    # replace the inference loop with this version to also collect probabilities and paths
    y_true, y_pred, paths = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            probs_all.extend(probs.tolist())
            # dataset returns samples in the same order as loader.dataset.samples
            # but we don't have paths from DataLoader; rebuild from indices:
            # safer: create a list of paths from ds.samples
        paths.extend([p for p, _ in ds.samples][len(paths):len(paths) + len(images)])

    # export CSV
    csv_path = exports_dir / f"test_predictions_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred"] + [f"prob_{c}" for c in classes])
        for i in range(len(y_true)):
            w.writerow([paths[i], classes[y_true[i]], classes[y_pred[i]]] + [f"{p:.6f}" for p in probs_all[i]])
    print("Saved per-image predictions:", csv_path)
