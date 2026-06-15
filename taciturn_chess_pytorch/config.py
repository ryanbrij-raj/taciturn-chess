"""
config.py — All hyperparameters for Taciturn Chess Engine (PyTorch).
"""

# ── Neural Network ────────────────────────────────────────────────────
NUM_RES_BLOCKS   = 10
NUM_FILTERS      = 128
L2_REG           = 1e-4

# ── MCTS ─────────────────────────────────────────────────────────────
MCTS_SIMULATIONS  = 100
MCTS_C_PUCT       = 1.5
DIRICHLET_ALPHA   = 0.3
DIRICHLET_EPSILON = 0.25

# ── Self-Play ─────────────────────────────────────────────────────────
SELF_PLAY_GAMES  = 25
MAX_GAME_MOVES   = 512
TEMP_THRESHOLD   = 30
TEMP_HIGH        = 1.0
TEMP_LOW         = 0.1

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE         = 256
EPOCHS_PER_ITER    = 5
LEARNING_RATE      = 0.001
LR_DECAY_STEPS     = [100, 200, 300]
REPLAY_BUFFER_SIZE = 500_000

# ── Evaluation ────────────────────────────────────────────────────────
EVAL_GAMES          = 40
WIN_RATE_THRESHOLD  = 0.55

# ── Paths ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints"
DATA_DIR        = "data"
LOG_DIR         = "logs"
BEST_MODEL_PATH = "checkpoints/best_model.pt"

# ── Board Encoding ────────────────────────────────────────────────────
HISTORY_LENGTH = 8
INPUT_CHANNELS = 119
POLICY_SIZE    = 4672
