"""
training/trainer.py — Train the neural network on self-play data (PyTorch).
"""

import torch
import torch.nn.functional as F
from config import BATCH_SIZE, EPOCHS_PER_ITER, LEARNING_RATE, LR_DECAY_STEPS, L2_REG
from utils.logger import get_logger

logger = get_logger("trainer")


def build_optimizer(model, iteration: int):
    lr = LEARNING_RATE
    decays = sum(1 for step in LR_DECAY_STEPS if iteration >= step)
    lr *= (0.1 ** decays)
    logger.info(f"Learning rate: {lr:.6f}")
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                           nesterov=True, weight_decay=L2_REG)


def train_on_buffer(model, device, replay_buffer, iteration: int) -> dict:
    if len(replay_buffer) < BATCH_SIZE:
        logger.warning(f"Buffer too small ({len(replay_buffer)} < {BATCH_SIZE}), skipping.")
        return {}

    model.train()
    optimizer = build_optimizer(model, iteration)
    steps_per_epoch = max(1, len(replay_buffer) // BATCH_SIZE)

    total_loss_acc = policy_loss_acc = value_loss_acc = 0.0
    n_steps = 0

    logger.info(
        f"Training: {EPOCHS_PER_ITER} epochs × {steps_per_epoch} steps "
        f"| buffer={len(replay_buffer):,}"
    )

    for epoch in range(EPOCHS_PER_ITER):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            boards, policies, values = replay_buffer.sample(BATCH_SIZE)

            boards_t   = torch.from_numpy(boards).float().to(device)
            policies_t = torch.from_numpy(policies).float().to(device)
            values_t   = torch.from_numpy(values).float().to(device)

            policy_logits, pred_values = model(boards_t)

            # Value loss: MSE
            value_loss = F.mse_loss(pred_values.squeeze(1), values_t)

            # policy loss: cross-entropy
            log_probs   = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_t * log_probs).sum(dim=1).mean()

            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_acc  += loss.item()
            policy_loss_acc += policy_loss.item()
            value_loss_acc  += value_loss.item()
            epoch_loss      += loss.item()
            n_steps += 1

        logger.info(f"  Epoch {epoch+1}/{EPOCHS_PER_ITER} | avg_loss={epoch_loss/steps_per_epoch:.4f}")

    model.eval()
    metrics = {
        "total_loss":  total_loss_acc  / n_steps,
        "policy_loss": policy_loss_acc / n_steps,
        "value_loss":  value_loss_acc  / n_steps,
    }
    logger.info(
        f"Training done | total={metrics['total_loss']:.4f} "
        f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
    )
    return metrics
