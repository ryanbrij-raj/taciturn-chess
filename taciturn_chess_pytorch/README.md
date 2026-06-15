# Taciturn Chess Engine — AlphaZero-Style

A self-learning chess engine inspired by DeepMind's AlphaZero. Uses deep reinforcement learning + Monte Carlo Tree Search (MCTS) to train entirely through self-play — no human games, no handcrafted heuristics.

---

## Architecture

```
Neural Network (ResNet)
  ├── Policy Head  → probability over 4672 possible moves
  └── Value Head   → estimated win probability [-1, +1]

MCTS
  └── Uses the neural net to guide tree search during self-play

Self-Play Loop
  ├── Generate games using MCTS + current best model
  ├── Store (board_state, policy, outcome) training examples
  └── Train new model → evaluate vs old → update if better
```

---

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA support (RTX recommended)
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.x

---

## Setup (Windows)

### 1. Install Python & CUDA
- Download Python 3.10+ from python.org
- Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Download cuDNN: https://developer.nvidia.com/cudnn

### 2. Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify GPU is detected
```powershell
python scripts\check_gpu.py
```

---

## Running

### Start self-play training
```powershell
python train.py
```

### Play against the engine (after training)
```powershell
python play.py
```

### Resume training from checkpoint
```powershell
python train.py --resume
```

### Configure training parameters
Edit `config.py` to adjust MCTS simulations, batch size, learning rate, etc.

---

## Training Pipeline

```
Iteration 1:
  Self-play 100 games with random/initial model
  → Train on generated data
  → Save checkpoint

Iteration 2:
  Self-play 100 games with new model
  → Evaluate new vs old (need 55%+ win rate to promote)
  → If better: replace best model
  → Repeat...
```

Training data and models are saved in `./data/` and `./checkpoints/`.

---

## Files

```
taciturn_chess/
├── config.py              # All hyperparameters
├── train.py               # Main training loop
├── play.py                # Human vs engine
├── requirements.txt
├── models/
│   ├── resnet.py          # Neural network architecture
│   └── checkpoint.py      # Save/load model weights
├── mcts/
│   ├── node.py            # MCTS tree node
│   └── search.py          # MCTS search algorithm
├── game/
│   ├── chess_game.py      # Board wrapper (python-chess)
│   └── encoder.py         # Board → tensor encoding
├── training/
│   ├── self_play.py       # Self-play game generation
│   ├── replay_buffer.py   # Training data storage
│   └── trainer.py         # Model training loop
└── utils/
    ├── logger.py           # Logging utilities
    └── evaluator.py        # Model vs model evaluation
```
