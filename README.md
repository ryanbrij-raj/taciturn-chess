# Taciturn Chess Engine

A self-learning chess engine inspired by DeepMind's AlphaZero. Built from scratch in Python with PyTorch, it learns to play chess entirely through self-play — no opening books, no handcrafted evaluation functions, no human game data. Just a neural network, Monte Carlo Tree Search, and millions of games against itself.

## How It Works

```
┌─────────────────────────────────────────────┐
│              Neural Network (ResNet)          │
│   ┌──────────────┐        ┌──────────────┐    │
│   │ Policy Head  │        │  Value Head  │    │
│   │ (best moves) │        │ (win chance) │    │
│   └──────────────┘        └──────────────┘    │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│        Monte Carlo Tree Search (MCTS)         │
│   Uses the network to guide which positions    │
│   are worth exploring deeper                    │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              Self-Play Training Loop            │
│  1. Play games against itself using MCTS        │
│  2. Record positions, move choices, outcomes    │
│  3. Train the network on what it learned        │
│  4. New model must beat old model to be promoted │
│  5. Repeat forever                              │
└─────────────────────────────────────────────┘
```

Every game the engine plays against itself becomes training data. Over hundreds of iterations, it gradually develops an intuition for piece value, board control, and tactics — the same way AlphaZero did, just at a much smaller scale.

## Features

- **Dual-headed ResNet** (~12.6M parameters) for move prediction and position evaluation
- **MCTS with PUCT** selection, Dirichlet noise for exploration, and temperature-based move sampling
- **Full chess rules** — castling, en passant, all four promotion types, 50-move rule, threefold repetition
- **Resumable training** — checkpoints and replay buffer persist across sessions
- **PGN export** — every self-play game is saved and viewable in any standard chess GUI or on Lichess
- **GPU-accelerated** via PyTorch + CUDA

## Requirements

- Windows 10/11 (or Linux/Mac with minor path tweaks)
- Python 3.11
- NVIDIA GPU with CUDA support
- ~5 GB disk space for dependencies

## Setup

```powershell
# Clone the repo
git clone https://github.com/ryanbrij-raj/taciturn-chess.git
cd taciturn-chess/taciturn_chess_pytorch

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install python-chess numpy tqdm matplotlib h5py

# Verify your GPU is detected
python scripts\check_gpu.py
```

## Usage

**Start training from scratch:**
```powershell
python train.py
```

**Resume training from your last checkpoint:**
```powershell
python train.py --resume
```

**Play against the engine:**
```powershell
python play.py            # You play White
python play.py --black    # You play Black
python play.py --sims 800 # Stronger engine, slower moves
```

Stop training anytime with `Ctrl+C` — progress is saved automatically.

## Project Structure

```
taciturn_chess_pytorch/
├── config.py              # All hyperparameters
├── train.py                # Main self-play training loop
├── play.py                 # Play against the engine
├── game/
│   ├── chess_game.py        # Game state wrapper
│   └── encoder.py           # Board ↔ tensor encoding
├── models/
│   ├── resnet.py             # Neural network architecture
│   └── checkpoint.py         # Save/load model weights
├── mcts/
│   ├── node.py                # Search tree node
│   └── search.py              # MCTS algorithm
├── training/
│   ├── self_play.py            # Self-play game generation
│   ├── replay_buffer.py        # Training data storage
│   └── trainer.py               # Gradient descent loop
└── utils/
    ├── logger.py                 # Logging
    └── evaluator.py               # Model vs. model evaluation
```

## How Strong Is It?

Strength scales directly with training time:

| Iterations | Approx. Strength |
|---|---|
| 1–10 | Random, no real strategy |
| 20–50 | Beginner |
| 100–200 | Casual club player |
| 500+ | Intermediate club player |

For reference, the original AlphaZero trained for hundreds of thousands of iterations across thousands of TPUs. This project is built for learning the architecture and watching an engine improve in real time — not for competing with engines like Stockfish or Leela Chess Zero.

## Tech Stack

Python · PyTorch · python-chess · Monte Carlo Tree Search · CUDA

## License

MIT
