import os, yaml
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import build_model
from src.gradcam import GradCAM

st.set_page_config(page_title='Leaf–Neck–Node Classifier', page_icon='🌿', layout='wide')

@st.cache_resource
def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model(weights_path: str, model_name: str, classes_path: str):
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    model, target_layer = build_model(model_name, num_classes=len(classes), pretrained=False)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, target_layer, classes

def preprocess(img: Image.Image, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tfm(img).unsqueeze(0)

def overlay_heatmap(image_bgr, heatmap, alpha=0.45):
    hm = (heatmap * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_bgr, 1 - alpha, hm, alpha, 0)

st.title('Leaf–Neck–Node Classifier (PyTorch + Grad-CAM)')
st.caption('Upload an image to classify and visualize Grad‑CAM. Configure dataset and parameters in config.yaml.')

cfg = load_cfg(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
classes_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classes.txt')

col1, col2 = st.columns(2)

with col1:
    st.subheader('1) Choose an image')
    uploaded = st.file_uploader('Upload JPG/PNG', type=['jpg','jpeg','png'])
    if uploaded is None:
        st.info('Upload an image after training is complete.')
        st.stop()
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Input image', use_column_width=True)

with col2:
    st.subheader('2) Prediction + Grad‑CAM')
    if not (os.path.exists(weights_path) and os.path.exists(classes_path)):
        st.warning('Weights not found. Train first to create models/best.pt and models/classes.txt.')
        st.stop()

    model_name = cfg.get('model_name', 'resnet18')
    img_size = cfg.get('img_size', 224)
    model, target_layer, classes = load_model(weights_path, model_name, classes_path)

    x = preprocess(img, img_size)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = classes[pred_idx]

    st.markdown(f'**Prediction:** `{pred_name}`')
    st.write({c: float(np.round(probs[i], 4)) for i, c in enumerate(classes)})

    cam = GradCAM(model, target_layer).generate(x, pred_idx).numpy()
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    overlay = overlay_heatmap(img_bgr, cam_resized, alpha=0.45)
    from PIL import Image as PILImage
    st.image(PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), caption='Grad‑CAM overlay', use_column_width=True)
