"""
utils/evaluator.py — Model vs model evaluation (PyTorch).
"""

import chess
from tqdm import tqdm

from game.chess_game import ChessGame
from mcts.search import MCTS
from config import EVAL_GAMES, WIN_RATE_THRESHOLD, MCTS_SIMULATIONS
from utils.logger import get_logger

logger = get_logger("evaluator")


def play_evaluation_game(white_model, black_model, device, num_simulations=None) -> float:
    if num_simulations is None:
        num_simulations = max(100, MCTS_SIMULATIONS // 2)
    game = ChessGame()
    white_mcts = MCTS(white_model, device)
    black_mcts = MCTS(black_model, device)

    while not game.is_terminal():
        mcts  = white_mcts if game.current_player == chess.WHITE else black_mcts
        move  = mcts.get_best_move(game.board, num_simulations)
        if move is None:
            break
        game.apply_move(move)
    return game.get_outcome()


def evaluate_models(new_model, best_model, device, num_games: int = EVAL_GAMES) -> dict:
    assert num_games % 2 == 0
    half = num_games // 2
    new_wins = draws = best_wins = 0

    logger.info(f"Evaluation: {num_games} games ({half} each color)")

    for _ in tqdm(range(half), desc="Eval (new=white)", unit="game"):
        o = play_evaluation_game(new_model, best_model, device)
        if o > 0:   new_wins  += 1
        elif o == 0: draws    += 1
        else:        best_wins += 1

    for _ in tqdm(range(half), desc="Eval (new=black)", unit="game"):
        o = play_evaluation_game(best_model, new_model, device)
        if o < 0:   new_wins  += 1
        elif o == 0: draws    += 1
        else:        best_wins += 1

    total    = new_wins + draws + best_wins
    win_rate = (new_wins + 0.5 * draws) / total if total > 0 else 0.0
    promoted = win_rate >= WIN_RATE_THRESHOLD

    result = dict(new_wins=new_wins, draws=draws, best_wins=best_wins,
                  win_rate=win_rate, promoted=promoted)
    logger.info(
        f"Evaluation: new={new_wins} draw={draws} best={best_wins} | "
        f"win_rate={win_rate:.1%} | {'✓ PROMOTED' if promoted else '✗ rejected'}"
    )
    return result
