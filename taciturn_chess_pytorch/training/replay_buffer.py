"""
training/replay_buffer.py — Ring buffer for storing self-play training examples.

Keeps the most recent N (board_tensor, policy, value) tuples.
Older data is automatically evicted as new data comes in.
Data is also saved to disk for resuming training.
"""

import os
import pickle
import numpy as np
from collections import deque
from config import REPLAY_BUFFER_SIZE, DATA_DIR
from utils.logger import get_logger

logger = get_logger("replay_buffer")
BUFFER_PATH = os.path.join(DATA_DIR, "replay_buffer.pkl")


class ReplayBuffer:
    """
    Fixed-size ring buffer of (board_tensor, policy, value) training examples.
    Thread-safe for single-process use.
    """

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, examples: list[tuple]):
        """Add a list of (tensor, policy, value) examples."""
        for ex in examples:
            self.buffer.append(ex)
        logger.info(f"Buffer size: {len(self.buffer):,} / {self.max_size:,}")

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch.
        
        Returns:
            boards:   (batch, 8, 8, INPUT_CHANNELS) float32
            policies: (batch, POLICY_SIZE) float32
            values:   (batch,) float32
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} examples, need {batch_size}"
            )
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        boards   = np.array([s[0] for s in samples], dtype=np.float32)
        policies = np.array([s[1] for s in samples], dtype=np.float32)
        values   = np.array([s[2] for s in samples], dtype=np.float32)
        return boards, policies, values

    def __len__(self):
        return len(self.buffer)

    def save(self):
        """Persist buffer to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Buffer saved → {BUFFER_PATH} ({len(self.buffer):,} examples)")

    def load(self):
        """Load buffer from disk if it exists."""
        if os.path.exists(BUFFER_PATH):
            with open(BUFFER_PATH, "rb") as f:
                self.buffer = pickle.load(f)
            logger.info(f"Buffer loaded ← {BUFFER_PATH} ({len(self.buffer):,} examples)")
        else:
            logger.info("No existing buffer found, starting fresh.")
