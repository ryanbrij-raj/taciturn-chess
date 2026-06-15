"""
mcts/search.py — MCTS with PyTorch neural network guidance.
"""

import numpy as np
import chess
from mcts.node import MCTSNode
from models.resnet import predict_single
from config import MCTS_C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON


class MCTS:
    def __init__(self, model, device, c_puct: float = MCTS_C_PUCT):
        self.model   = model
        self.device  = device
        self.c_puct  = c_puct

    def search(self, board: chess.Board, num_simulations: int, add_noise: bool = True):
        root = MCTSNode(board.copy())
        self._expand_node(root)
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            node = root
            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child(self.c_puct)
            if not node.is_terminal():
                value = self._expand_node(node)
            else:
                value = self._terminal_value(node)
            self._backup(node, value)

        from config import POLICY_SIZE
        from game.encoder import move_to_index
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for move, child in root.children.items():
            policy[move_to_index(move)] = child.visit_count
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy, root.q_value

    def get_best_move(self, board: chess.Board, num_simulations: int):
        root = MCTSNode(board.copy())
        self._expand_node(root)
        for _ in range(num_simulations):
            node = root
            while not node.is_leaf() and not node.is_terminal():
                node = node.best_child(self.c_puct)
            value = self._expand_node(node) if not node.is_terminal() else self._terminal_value(node)
            self._backup(node, value)
        if not root.children:
            return None
        return max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]

    def _expand_node(self, node: MCTSNode) -> float:
        from game.encoder import board_to_tensor
        tensor = board_to_tensor(node.board)
        policy_probs, value = predict_single(self.model, tensor, self.device)
        node.expand(policy_probs)
        return value

    def _terminal_value(self, node: MCTSNode) -> float:
        result = node.board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if node.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return 1.0 if node.board.turn == chess.BLACK else -1.0
        return 0.0

    def _backup(self, node: MCTSNode, value: float):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, root: MCTSNode):
        n = len(root.children)
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
        for child, eps in zip(root.children.values(), noise):
            child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * eps


def sample_move(policy: np.ndarray, legal_moves: list, temperature: float = 1.0):
    from game.encoder import move_to_index
    probs = np.array([policy[move_to_index(m)] for m in legal_moves], dtype=np.float64)
    if probs.sum() == 0:
        probs = np.ones(len(legal_moves)) / len(legal_moves)
    else:
        probs /= probs.sum()
    if temperature < 0.01:
        return legal_moves[int(np.argmax(probs))]
    probs = probs ** (1.0 / temperature)
    probs /= probs.sum()
    return legal_moves[np.random.choice(len(legal_moves), p=probs)]
