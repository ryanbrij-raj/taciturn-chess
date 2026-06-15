"""
game/chess_game.py — Thin wrapper around python-chess for the engine.
"""

import chess
import chess.pgn
from game.encoder import board_to_tensor, legal_moves_mask
from config import MAX_GAME_MOVES


class ChessGame:
    """Manages a single chess game with history tracking."""

    def __init__(self):
        self.board = chess.Board()
        self.history: list[chess.Board] = []  # previous board states
        self.move_history: list[chess.Move] = []

    def reset(self):
        self.board = chess.Board()
        self.history = []
        self.move_history = []

    def clone(self) -> "ChessGame":
        g = ChessGame()
        g.board = self.board.copy()
        g.history = [b.copy() for b in self.history]
        g.move_history = list(self.move_history)
        return g

    # state accessors

    @property
    def current_player(self) -> chess.Color:
        return self.board.turn

    def get_tensor(self):
        """Return current board as (8, 8, INPUT_CHANNELS) float32 tensor."""
        return board_to_tensor(self.board, self.history)

    def get_legal_mask(self):
        """Return boolean mask of legal moves over policy space."""
        return legal_moves_mask(self.board)

    def get_legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    # game flow

    def apply_move(self, move: chess.Move):
        self.history.insert(0, self.board.copy())
        if len(self.history) > 7:
            self.history = self.history[:7]
        self.move_history.append(move)
        self.board.push(move)

    def is_terminal(self) -> bool:
        if self.board.is_game_over(claim_draw=True):
            return True
        if len(self.move_history) >= MAX_GAME_MOVES:
            return True
        return False

    def get_outcome(self) -> float:
        """
        Returns outcome from the perspective of WHITE:
          +1.0  = white wins
          -1.0  = black wins
           0.0  = draw / too long
        """
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0  # draw or ongoing (treated as draw if terminal)

    def get_outcome_for_player(self, color: chess.Color) -> float:
        """Returns outcome from the given player's perspective."""
        raw = self.get_outcome()
        return raw if color == chess.WHITE else -raw

    # utilities

    def to_pgn(self) -> str:
        """Export the game to PGN format."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Taciturn Self-Play"
        node = game
        board = chess.Board()
        for move in self.move_history:
            node = node.add_variation(move)
            board.push(move)
        return str(game)

    def __repr__(self):
        return f"ChessGame(ply={self.board.ply()}, turn={'W' if self.board.turn else 'B'})"
