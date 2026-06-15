"""
mcts/node.py — MCTS tree node.

Each node represents a game state. Edges represent moves.
Stores visit counts N, total action values W, and prior priors P.
"""

import math
import chess
import numpy as np


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    
    Attributes:
        state:      chess.Board at this node
        parent:     parent MCTSNode (None for root)
        move:       chess.Move that led here from parent
        children:   dict[chess.Move → MCTSNode]
        prior:      P(s,a) — prior probability from neural net
        visit_count: N(s,a)
        value_sum:  W(s,a) — sum of backpropagated values
        is_expanded: whether we've expanded this node's children
    """

    __slots__ = [
        "board", "parent", "move", "children",
        "prior", "visit_count", "value_sum", "is_expanded"
    ]

    def __init__(
        self,
        board: chess.Board,
        parent: "MCTSNode" = None,
        move: chess.Move = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: dict[chess.Move, MCTSNode] = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    # value estimates

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a) = W / N"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """
        PUCT score used by AlphaZero:
          Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
        """
        parent_visits = self.parent.visit_count if self.parent else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    # tree operations

    def expand(self, policy_probs: np.ndarray):
        """
        Create child nodes for all legal moves.
        
        Args:
            policy_probs: (POLICY_SIZE,) prior probabilities from neural net
        """
        from game.encoder import move_to_index
        self.is_expanded = True

        legal = list(self.board.legal_moves)
        if not legal:
            return

        # Gather priors for legal moves
        move_priors = {}
        total = 0.0
        for move in legal:
            idx = move_to_index(move)
            p = float(policy_probs[idx])
            move_priors[move] = p
            total += p

        # normalize (in case policy didn't sum to 1 over legal moves)
        if total > 0:
            for move in legal:
                move_priors[move] /= total
        else:
            # Uniform prior as fallback
            for move in legal:
                move_priors[move] = 1.0 / len(legal)

        for move in legal:
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(
                board=child_board,
                parent=self,
                move=move,
                prior=move_priors[move],
            )

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def update(self, value: float):
        """Backpropagate value through this node."""
        self.visit_count += 1
        self.value_sum += value

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def __repr__(self):
        return (
            f"MCTSNode(move={self.move}, N={self.visit_count}, "
            f"Q={self.q_value:.3f}, P={self.prior:.3f})"
        )
