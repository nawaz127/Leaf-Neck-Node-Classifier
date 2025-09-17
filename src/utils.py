import os, random, yaml
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
