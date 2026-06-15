"""
play.py — Play against the Taciturn Chess Engine (PyTorch).

Usage:
  python play.py              # You play White
  python play.py --black      # You play Black
  python play.py --sims 800   # More MCTS simulations
"""

import argparse
import chess
import torch

from models.checkpoint import load_best_model, model_exists
from models.resnet import build_model
from game.chess_game import ChessGame
from mcts.search import MCTS
from config import MCTS_SIMULATIONS, INPUT_CHANNELS

BANNER = """
╔══════════════════════════════════════╗
║   TACITURN CHESS ENGINE              ║
║   AlphaZero-style self-play engine   ║
╚══════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  <move>    Make a move (e.g. e2e4, Nf3, O-O)
  board     Reprint the board
  undo      Take back last 2 moves
  resign    Forfeit the game
  quit      Exit
"""


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_board(board, human_color):
    flip = (human_color == chess.BLACK)
    print()
    ranks = range(7, -1, -1) if not flip else range(0, 8)
    files = range(0, 8)      if not flip else range(7, -1, -1)
    file_labels = "abcdefgh" if not flip else "hgfedcba"
    for r in ranks:
        row = ""
        for f in files:
            piece = board.piece_at(chess.square(f, r))
            row  += f" {piece.unicode_symbol() if piece else '·'}"
        print(f"  {r+1} {row}")
    print(f"    {' '.join(file_labels)}\n")


def parse_move(board, move_str):
    move_str = move_str.strip()
    try:
        m = chess.Move.from_uci(move_str)
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    try:
        m = board.parse_san(move_str)
        if m in board.legal_moves:
            return m
    except Exception:
        pass
    return None


def main(human_color, num_simulations):
    print(BANNER)
    device = get_device()
    print(f"Device: {device}")

    if not model_exists():
        print("⚠  No trained model found. Run 'python train.py' first.")
        print("   Starting with untrained model for demo...\n")
        model = build_model(device)
    else:
        print("Loading model...")
        model = load_best_model(device)
        print("✓ Model loaded\n")

    model.eval()
    mcts = MCTS(model, device)
    game = ChessGame()

    print(f"You are playing as {'White' if human_color == chess.WHITE else 'Black'}.")
    print(f"Engine: {num_simulations} MCTS simulations per move.")
    print(HELP_TEXT)
    print_board(game.board, human_color)

    while not game.is_terminal():
        if game.current_player == human_color:
            while True:
                try:
                    cmd = input("Your move: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye."); return

                if cmd in ("quit", "q", "exit"):
                    print("Goodbye."); return
                elif cmd == "board":
                    print_board(game.board, human_color); continue
                elif cmd == "help":
                    print(HELP_TEXT); continue
                elif cmd == "resign":
                    winner = "Black" if human_color == chess.WHITE else "White"
                    print(f"You resigned. {winner} wins."); return
                elif cmd == "undo":
                    if len(game.move_history) >= 2:
                        game.board.pop(); game.board.pop()
                        game.move_history = game.move_history[:-2]
                        game.history = game.history[2:] if len(game.history) >= 2 else []
                        print("Moves undone.")
                        print_board(game.board, human_color)
                    else:
                        print("Nothing to undo.")
                    continue

                move = parse_move(game.board, cmd)
                if move is None:
                    examples = [game.board.san(m) for m in list(game.board.legal_moves)[:10]]
                    print(f"Illegal move. Examples: {', '.join(examples)}")
                    continue
                game.apply_move(move)
                print_board(game.board, human_color)
                break
        else:
            print(f"Engine thinking ({num_simulations} sims)...", end="", flush=True)
            move = mcts.get_best_move(game.board, num_simulations)
            if move is None:
                print(" no legal moves"); break
            san = game.board.san(move)
            game.apply_move(move)
            print(f"\rEngine plays: {san}           ")
            print_board(game.board, human_color)

    result  = game.board.result(claim_draw=True)
    outcome = game.get_outcome()
    print(f"\nGame over! Result: {result}")
    if outcome == 0:
        print("It's a draw.")
    elif (outcome > 0) == (human_color == chess.WHITE):
        print("You win! 🎉")
    else:
        print("Engine wins.")
    print("\nFull game PGN:")
    print(game.to_pgn())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--black", action="store_true")
    parser.add_argument("--sims", type=int, default=MCTS_SIMULATIONS)
    args = parser.parse_args()
    main(chess.BLACK if args.black else chess.WHITE, args.sims)
