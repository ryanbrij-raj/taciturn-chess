"""
models/checkpoint.py — Save and load PyTorch model weights.
"""

import os
import torch
from models.resnet import build_model, TaciturnNet
from config import CHECKPOINT_DIR, BEST_MODEL_PATH
from utils.logger import get_logger

logger = get_logger("checkpoint")


def save_model(model: TaciturnNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model → {path}")


def load_model(path: str, device: torch.device) -> TaciturnNet:
    model = build_model(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    logger.info(f"Loaded model ← {path}")
    return model


def save_checkpoint(model: TaciturnNet, iteration: int):
    path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt")
    save_model(model, path)
    return path


def load_best_model(device: torch.device) -> TaciturnNet:
    if os.path.exists(BEST_MODEL_PATH):
        return load_model(BEST_MODEL_PATH, device)
    logger.info("No best model found — creating fresh model.")
    return build_model(device)


def save_best_model(model: TaciturnNet):
    save_model(model, BEST_MODEL_PATH)


def model_exists() -> bool:
    return os.path.exists(BEST_MODEL_PATH)
