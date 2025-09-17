# Leaf–Neck–Node Classifier (PyTorch + Grad-CAM + Streamlit)

Edit `config.yaml` to point to your dataset, then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config config.yaml
streamlit run app/streamlit_app.py
```
