import os, argparse, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .utils import set_seed, device, load_config
from .data import get_dataloaders
from .model import build_model

def train_one_epoch(model, loader, criterion, optimizer, scaler, dev):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(dev), labels.to(dev)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, dev):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(dev), labels.to(dev)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    dev = device()

    train_loader, val_loader, class_names = get_dataloaders(
        cfg["data_dir"], cfg.get("batch_size", 32), cfg.get("img_size", 224)
    )
    model, _ = build_model(cfg.get("model_name","resnet18"), num_classes=len(class_names), pretrained=True)
    model.to(dev)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=cfg.get("learning_rate", 3e-4), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 15))
    scaler = torch.cuda.amp.GradScaler() if (cfg.get("use_amp", True) and torch.cuda.is_available()) else None

    os.makedirs("models", exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, cfg.get("epochs", 15)+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, dev)
        val_loss, val_acc = evaluate(model, val_loader, criterion, dev)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=val_loss, val_acc=val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best.pt")
            print(f"  -> Saved new best to models/best.pt (val_acc={best_acc:.4f})")

    with open("models/classes.txt", "w") as f:
        for c in class_names:
            f.write(c+"\n")
    os.makedirs("exports", exist_ok=True)
    with open("exports/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete.")

if __name__ == "__main__":
    main()
