"""
game/encoder.py — Board → tensor encoding (AlphaZero format, PyTorch-friendly).

Output tensor shape: (INPUT_CHANNELS, 8, 8) float32  [channels-first for PyTorch]
"""

import numpy as np
import chess
from config import HISTORY_LENGTH, INPUT_CHANNELS, POLICY_SIZE

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

QUEEN_DIRS   = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
KNIGHT_MOVES = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
UNDERPROM_DIRS   = [-1, 0, 1]
UNDERPROM_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]


def _build_move_index():
    move_to_idx = {}
    idx_to_move = {}
    idx = 0
    for sq in chess.SQUARES:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        for df, dr in QUEEN_DIRS:
            for dist in range(1, 8):
                tf, tr = f + df*dist, r + dr*dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    uci = chess.Move(sq, chess.square(tf, tr)).uci()
                    if uci not in move_to_idx:
                        move_to_idx[uci] = idx
                        idx_to_move[idx] = uci
                        idx += 1
        for df, dr in KNIGHT_MOVES:
            tf, tr = f+df, r+dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                uci = chess.Move(sq, chess.square(tf, tr)).uci()
                if uci not in move_to_idx:
                    move_to_idx[uci] = idx
                    idx_to_move[idx] = uci
                    idx += 1
        if r == 6:
            for df in UNDERPROM_DIRS:
                tf, tr = f+df, r+1
                if 0 <= tf < 8 and tr == 7:
                    for piece in UNDERPROM_PIECES:
                        uci = chess.Move(sq, chess.square(tf,tr), promotion=piece).uci()
                        if uci not in move_to_idx:
                            move_to_idx[uci] = idx
                            idx_to_move[idx] = uci
                            idx += 1
        if r == 1:
            for df in UNDERPROM_DIRS:
                tf, tr = f+df, r-1
                if 0 <= tf < 8 and tr == 0:
                    for piece in UNDERPROM_PIECES:
                        uci = chess.Move(sq, chess.square(tf,tr), promotion=piece).uci()
                        if uci not in move_to_idx:
                            move_to_idx[uci] = idx
                            idx_to_move[idx] = uci
                            idx += 1
    return move_to_idx, idx_to_move


MOVE_TO_IDX, IDX_TO_MOVE = _build_move_index()


def move_to_index(move: chess.Move) -> int:
    if move.promotion == chess.QUEEN:
        move = chess.Move(move.from_square, move.to_square)
    return MOVE_TO_IDX.get(move.uci(), 0)


def index_to_move(idx: int) -> chess.Move:
    uci = IDX_TO_MOVE.get(idx)
    return chess.Move.from_uci(uci) if uci else None


def board_to_tensor(board: chess.Board, history: list = None) -> np.ndarray:
    """
    Returns (INPUT_CHANNELS, 8, 8) float32 — channels first for PyTorch.
    """
    planes = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float32)
    boards = [board] + (list(reversed(history)) if history else [])
    boards = boards[:HISTORY_LENGTH]
    flip = (board.turn == chess.BLACK)

    for t, b in enumerate(boards):
        base = t * 14
        for sq in chess.SQUARES:
            piece = b.piece_at(sq)
            if piece is None:
                continue
            r, f = divmod(sq, 8)
            if flip:
                r = 7 - r
            plane_idx = PIECE_TO_PLANE.get((piece.piece_type, piece.color))
            if plane_idx is not None:
                if flip:
                    plane_idx = (plane_idx + 6) % 12
                planes[base + plane_idx, r, f] = 1.0
        if t == 0:
            if b.is_repetition(2):
                planes[base + 12, :, :] = 1.0
            if b.is_repetition(3):
                planes[base + 13, :, :] = 1.0

    planes[112, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[113, :, :] = board.fullmove_number / 500.0
    planes[114, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[115, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[116, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[117, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    planes[118, :, :] = board.halfmove_clock / 100.0
    return planes


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
