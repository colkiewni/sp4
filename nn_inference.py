"""
ONNX inference wrapper. CPU-only, fast.
Used at play time after models are trained and exported.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

from engine import Card, DealState, GameState, legal_plays
from features import encode_state, encode_hand_for_bidding, CARD_INDEX


class PlayModelInference:
    """Wraps play_model.onnx for fast CPU inference."""

    def __init__(self, model_path: str):
        assert _ORT_AVAILABLE, "onnxruntime not installed: pip install onnxruntime"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name

    def policy_and_value(
        self,
        deal: DealState,
        game: GameState,
        player: int
    ) -> tuple[dict[Card, float], float]:
        """
        Returns (policy_dict, value).
        policy_dict: legal cards → probability
        value: float in [-1, 1]
        """
        feat = encode_state(deal, game, player).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: feat})
        policy_logits = outputs[0][0]  # (52,)
        value = float(outputs[1][0][0])

        # Mask illegal moves
        legal = legal_plays(deal, player)
        legal_indices = [CARD_INDEX[c] for c in legal]

        # Softmax over legal moves only
        logits = policy_logits[legal_indices]
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        policy = {card: float(prob) for card, prob in zip(legal, probs)}
        return policy, value

    def value_only(self, deal: DealState, game: GameState, team: int) -> float:
        """Quick value estimate from current player's perspective."""
        player = deal.current_player
        if player < 0:
            return 0.0
        _, value = self.policy_and_value(deal, game, player)
        # Flip sign if current player is on the other team
        if deal.team_of(player) != team:
            value = -value
        return value


class BidModelInference:
    """Wraps bid_model.onnx for fast CPU inference."""

    def __init__(self, model_path: str):
        assert _ORT_AVAILABLE, "onnxruntime not installed: pip install onnxruntime"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name

    def predict_bid(
        self,
        hand,
        player: int,
        deal: DealState,
        game: GameState
    ) -> int:
        """Return the model's best bid (0-13)."""
        feat = encode_hand_for_bidding(hand, player, deal, game).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: feat})
        logits = outputs[0][0]  # (14,)
        return int(np.argmax(logits))


# Global singletons — loaded once, reused
_play_model: Optional[PlayModelInference] = None
_bid_model: Optional[BidModelInference] = None


def load_models(play_path: str = 'models/play_model.onnx',
                bid_path: str = 'models/bid_model.onnx'):
    """Load models into global singletons. Call once at startup."""
    global _play_model, _bid_model
    import os
    if os.path.exists(play_path):
        _play_model = PlayModelInference(play_path)
        print(f"Loaded play model: {play_path}")
    if os.path.exists(bid_path):
        _bid_model = BidModelInference(bid_path)
        print(f"Loaded bid model: {bid_path}")


def get_play_model() -> Optional[PlayModelInference]:
    return _play_model


def get_bid_model() -> Optional[BidModelInference]:
    return _bid_model