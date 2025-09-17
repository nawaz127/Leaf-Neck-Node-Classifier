import argparse, torch
from PIL import Image
import numpy as np
from torchvision import transforms
from .model import build_model

def load_classes(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def preprocess(img: Image.Image, img_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(img).unsqueeze(0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--weights", type=str, default="models/best.pt")
    p.add_argument("--classes", type=str, default="models/classes.txt")
    args = p.parse_args()

    classes = load_classes(args.classes)
    m, _ = build_model(args.model, num_classes=len(classes), pretrained=False)
    m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    m.eval()

    img = Image.open(args.img).convert("RGB")
    x = preprocess(img, args.img_size)
    with torch.no_grad():
        logits = m(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = int(np.argmax(probs))
    print(f"Pred: {classes[pred_idx]}  Probs: {[round(float(p),4) for p in probs]}")

if __name__ == "__main__":
    main()
