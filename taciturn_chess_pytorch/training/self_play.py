"""
training/self_play.py — Generate training data through self-play (PyTorch).
Saves every game as a PGN file in the games/ directory.
"""

import os
import chess
import chess.pgn
import numpy as np
from datetime import datetime
from tqdm import tqdm

from game.chess_game import ChessGame
from game.encoder import board_to_tensor
from mcts.search import MCTS, sample_move
from config import SELF_PLAY_GAMES, MCTS_SIMULATIONS, TEMP_THRESHOLD, TEMP_HIGH, TEMP_LOW
from utils.logger import get_logger

logger = get_logger("self_play")

GAMES_DIR = "games"


def save_pgn(game: ChessGame, iteration: int, game_num: int, result: str):
    """Save a single game to a PGN file."""
    os.makedirs(GAMES_DIR, exist_ok=True)

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"]  = f"Taciturn Self-Play Iter {iteration}"
    pgn_game.headers["Site"]   = "Local"
    pgn_game.headers["Date"]   = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"]  = str(game_num)
    pgn_game.headers["White"]  = f"Taciturn-Iter{iteration}"
    pgn_game.headers["Black"]  = f"Taciturn-Iter{iteration}"
    pgn_game.headers["Result"] = result

    node = pgn_game
    for move in game.move_history:
        node = node.add_variation(move)

    # One file per iteration, all games appended inside it
    path = os.path.join(GAMES_DIR, f"iter_{iteration:04d}.pgn")
    with open(path, "a", encoding="utf-8") as f:
        print(pgn_game, file=f)
        print(file=f)  # blank line between games


class GameRecord:
    def __init__(self):
        self.board_tensors = []
        self.policies      = []
        self.players       = []

    def add_step(self, tensor, policy, player):
        self.board_tensors.append(tensor)
        self.policies.append(policy)
        self.players.append(player)

    def finalize(self, outcome: float):
        examples = []
        for tensor, policy, player in zip(self.board_tensors, self.policies, self.players):
            value = outcome if player == chess.WHITE else -outcome
            examples.append((tensor, policy, value))
        return examples


def play_one_game(model, device, iteration: int, game_num: int) -> tuple:
    game   = ChessGame()
    mcts   = MCTS(model, device)
    record = GameRecord()
    move_count = 0

    while not game.is_terminal():
        temp   = TEMP_HIGH if move_count < TEMP_THRESHOLD else TEMP_LOW
        policy, _ = mcts.search(game.board, MCTS_SIMULATIONS, add_noise=True)
        record.add_step(game.get_tensor(), policy, game.current_player)
        legal = game.get_legal_moves()
        if not legal:
            break
        move = sample_move(policy, legal, temperature=temp)
        game.apply_move(move)
        move_count += 1

    outcome    = game.get_outcome()
    result_str = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}.get(outcome, "*")

    # Save PGN
    save_pgn(game, iteration, game_num, result_str)

    return record.finalize(outcome), result_str


def generate_self_play_data(model, device, num_games: int = SELF_PLAY_GAMES, iteration: int = 1) -> list:
    all_examples = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    logger.info(f"Starting self-play: {num_games} games, {MCTS_SIMULATIONS} sims/move")
    logger.info(f"PGNs will be saved to: {os.path.abspath(GAMES_DIR)}/iter_{iteration:04d}.pgn")

    for i in tqdm(range(num_games), desc="Self-play", unit="game"):
        examples, result = play_one_game(model, device, iteration, i + 1)
        all_examples.extend(examples)
        results[result] = results.get(result, 0) + 1

    logger.info(
        f"Self-play complete: {len(all_examples)} examples | "
        f"W={results['1-0']} D={results['1/2-1/2']} L={results['0-1']}"
    )
    return all_examples