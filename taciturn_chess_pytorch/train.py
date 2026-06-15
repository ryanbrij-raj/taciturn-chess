"""
train.py — Main training loop for Taciturn Chess Engine (PyTorch).

Usage:
  python train.py           # start fresh
  python train.py --resume  # resume from last checkpoint
"""

import os
import sys
import argparse
import json
import copy
from datetime import datetime

import torch

from models.resnet import build_model
from models.checkpoint import save_checkpoint, save_best_model, load_best_model, model_exists
from training.self_play import generate_self_play_data
from training.replay_buffer import ReplayBuffer
from training.trainer import train_on_buffer
from utils.evaluator import evaluate_models
from utils.logger import get_logger
from config import CHECKPOINT_DIR, DATA_DIR, LOG_DIR, SELF_PLAY_GAMES

logger = get_logger("train")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem    = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {name} ({mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("WARNING: No GPU detected. Training on CPU will be slow.")
        logger.info("Run: python scripts/check_gpu.py  for diagnostics.")
    return device


def load_training_state() -> int:
    path = os.path.join(DATA_DIR, "training_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("iteration", 0)
    return 0


def save_training_state(iteration: int, metrics: dict = None):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "training_state.json")
    with open(path, "w") as f:
        json.dump({"iteration": iteration,
                   "timestamp": datetime.now().isoformat(),
                   "metrics": metrics or {}}, f, indent=2)


def plot_training_curve(history: list):
    try:
        import matplotlib.pyplot as plt
        iters    = [h["iteration"]           for h in history]
        losses   = [h.get("total_loss",  0)  for h in history]
        values   = [h.get("value_loss",  0)  for h in history]
        policies = [h.get("policy_loss", 0)  for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(iters, losses);   axes[0].set_title("Total Loss")
        axes[1].plot(iters, values);   axes[1].set_title("Value Loss")
        axes[2].plot(iters, policies); axes[2].set_title("Policy Loss")
        for ax in axes:
            ax.set_xlabel("Iteration"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, "training_curve.png")
        plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Training curve → {path}")
    except Exception as e:
        logger.warning(f"Could not plot: {e}")


def main(resume: bool = False):
    for d in (CHECKPOINT_DIR, DATA_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

    device = get_device()

    if resume and model_exists():
        logger.info("Resuming training...")
        best_model = load_best_model(device)
        start_iter = load_training_state()
    else:
        logger.info("Starting fresh training run.")
        best_model = build_model(device)
        params = sum(p.numel() for p in best_model.parameters())
        logger.info(f"Model: {params:,} parameters")
        save_best_model(best_model)
        start_iter = 0

    replay_buffer = ReplayBuffer()
    if resume:
        replay_buffer.load()

    history   = []
    iteration = start_iter + 1

    logger.info("=" * 60)
    logger.info("  Taciturn Chess Engine — Self-Play Training (PyTorch)")
    logger.info("=" * 60)

    while True:
        logger.info(f"\n{'='*50}\n  ITERATION {iteration}\n{'='*50}")

        # 1. Self-play
        logger.info(f"[1/4] Generating {SELF_PLAY_GAMES} self-play games...")
        examples = generate_self_play_data(best_model, device, num_games=SELF_PLAY_GAMES, iteration=iteration)
        replay_buffer.add(examples)

        # 2. Train new model (copy weights from best as starting point)
        logger.info("[2/4] Training new model...")
        new_model = build_model(device)
        new_model.load_state_dict(copy.deepcopy(best_model.state_dict()))
        metrics = train_on_buffer(new_model, device, replay_buffer, iteration)

        # 3. Evaluate
        logger.info("[3/4] Evaluating new model vs best model...")
        eval_result = evaluate_models(new_model, best_model, device)

        # 4. Promote if better
        if eval_result["promoted"]:
            logger.info(f"[4/4] ✓ New model PROMOTED (win rate: {eval_result['win_rate']:.1%})")
            best_model = new_model
            save_best_model(best_model)
        else:
            logger.info(f"[4/4] ✗ New model rejected (win rate: {eval_result['win_rate']:.1%})")

        save_checkpoint(best_model, iteration)
        replay_buffer.save()

        record = {"iteration": iteration, **metrics,
                  **{f"eval_{k}": v for k, v in eval_result.items()}}
        history.append(record)
        save_training_state(iteration, record)
        plot_training_curve(history)

        logger.info(f"Iteration {iteration} complete.")
        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(resume=args.resume)
