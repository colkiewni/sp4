"""
State → feature vector conversion for neural network input.
"""

from __future__ import annotations
import numpy as np
from engine import Card, Suit, Rank, DealState, GameState, DECK

# Fixed card ordering for feature vectors
CARD_INDEX = {card: i for i, card in enumerate(DECK)}
NUM_CARDS = 52
FEATURE_DIM = 370  # total feature vector size


def card_to_vec(cards, size=NUM_CARDS) -> np.ndarray:
    """Binary vector indicating which cards are present."""
    v = np.zeros(size, dtype=np.float32)
    for c in cards:
        v[CARD_INDEX[c]] = 1.0
    return v


def trick_to_vec(trick_cards: list[tuple[int, Card]], perspective: int) -> np.ndarray:
    """
    Encode current trick cards relative to perspective player.
    4 slots × 52 = 208 dims. Each slot = one position relative to player.
    """
    v = np.zeros(4 * NUM_CARDS, dtype=np.float32)
    for player, card in trick_cards:
        rel = (player - perspective) % 4
        v[rel * NUM_CARDS + CARD_INDEX[card]] = 1.0
    return v


def encode_state(deal: DealState, game: GameState, player: int) -> np.ndarray:
    """
    Encode full game state from player's perspective.
    Returns flat feature vector of size FEATURE_DIM.
    """
    features = []

    # 1. My hand (52)
    features.append(card_to_vec(deal.hands[player]))

    # 2. All cards played so far (52)
    features.append(card_to_vec(deal.cards_played))

    # 3. Current trick (4 × 52 = 208)
    trick_cards = deal.current_trick.cards if deal.current_trick else []
    features.append(trick_to_vec(trick_cards, player))

    # 4. My position relative to dealer (4) one-hot
    rel_pos = (player - deal.dealer) % 4
    pos_vec = np.zeros(4, dtype=np.float32)
    pos_vec[rel_pos] = 1.0
    features.append(pos_vec)

    # 5. Trick number (13) one-hot
    trick_vec = np.zeros(13, dtype=np.float32)
    if 0 <= deal.trick_number < 13:
        trick_vec[deal.trick_number] = 1.0
    features.append(trick_vec)

    # 6. Lead position relative to me (4) one-hot
    lead_vec = np.zeros(4, dtype=np.float32)
    if deal.current_trick and deal.current_trick.cards:
        rel_lead = (deal.current_trick.lead_player - player) % 4
        lead_vec[rel_lead] = 1.0
    features.append(lead_vec)

    # 7. Bids - relative order (4): self, left, partner, right
    bid_vec = np.zeros(4, dtype=np.float32)
    for i in range(4):
        p = (player + i) % 4
        bid = deal.bids[p]
        bid_vec[i] = bid / 13.0 if bid >= 0 else -0.1
    features.append(bid_vec)

    # 8. Tricks won - relative order (4)
    tricks_vec = np.zeros(4, dtype=np.float32)
    for i in range(4):
        p = (player + i) % 4
        tricks_vec[i] = deal.tricks_won[p] / 13.0
    features.append(tricks_vec)

    # 9. Scores normalized (2): my team, opponent team
    team = deal.team_of(player)
    target = max(game.target_score, 1)
    score_vec = np.array([
        game.scores[team] / target,
        game.scores[1 - team] / target
    ], dtype=np.float32)
    features.append(score_vec)

    # 10. Bags normalized (2)
    bags_vec = np.array([
        game.bags[team] / 10.0,
        game.bags[1 - team] / 10.0
    ], dtype=np.float32)
    features.append(bags_vec)

    # 11. Spades broken (1)
    features.append(np.array([float(deal.spades_broken)], dtype=np.float32))

    # 12. Void matrix (4 players × 4 suits = 16)
    void_vec = np.zeros(16, dtype=np.float32)
    for trick in deal.trick_history:
        lead_suit = trick.lead_suit
        leader = trick.lead_player
        for p, card in trick.cards:
            if p != leader and card.suit != lead_suit:
                rel_p = (p - player) % 4
                void_vec[rel_p * 4 + int(lead_suit)] = 1.0
    if deal.current_trick and deal.current_trick.cards:
        lead_suit = deal.current_trick.lead_suit
        leader = deal.current_trick.lead_player
        for p, card in deal.current_trick.cards:
            if p != leader and card.suit != lead_suit:
                rel_p = (p - player) % 4
                void_vec[rel_p * 4 + int(lead_suit)] = 1.0
    features.append(void_vec)

    # 13. Nil flags (4): relative order
    nil_vec = np.zeros(4, dtype=np.float32)
    for i in range(4):
        p = (player + i) % 4
        if deal.bids[p] == 0:
            nil_vec[i] = 1.0
    features.append(nil_vec)

    # 14. Team bids (2): my team total, opponent team total
    team_bids = np.zeros(2, dtype=np.float32)
    for p in range(4):
        t = deal.team_of(p)
        b = deal.bids[p]
        if b > 0:
            if t == team:
                team_bids[0] += b
            else:
                team_bids[1] += b
    team_bids /= 13.0
    features.append(team_bids)

    # 15. Tricks still needed by each team (2)
    needed = np.zeros(2, dtype=np.float32)
    for t in range(2):
        real_t = team if t == 0 else (1 - team)
        tb = sum(deal.bids[p] for p in range(4) if deal.team_of(p) == real_t and deal.bids[p] > 0)
        tw = sum(deal.tricks_won[p] for p in range(4) if deal.team_of(p) == real_t)
        needed[t] = max(0, tb - tw) / 13.0
    features.append(needed)

    result = np.concatenate(features)
    assert result.shape[0] == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {result.shape[0]}"
    return result


def encode_hand_for_bidding(hand: set[Card], player: int, deal: DealState, game: GameState) -> np.ndarray:
    """
    Encode state for bidding model.
    Simpler than play encoding: just hand + position + scores + partner bid.
    """
    features = []

    # Hand (52)
    features.append(card_to_vec(hand))

    # Position relative to dealer (4)
    rel_pos = (player - deal.dealer) % 4
    pos_vec = np.zeros(4, dtype=np.float32)
    pos_vec[rel_pos] = 1.0
    features.append(pos_vec)

    # Partner bid if known (2): [bid/13, known_flag]
    partner = (player + 2) % 4
    pb = deal.bids[partner]
    if pb >= 0:
        features.append(np.array([pb / 13.0, 1.0], dtype=np.float32))
    else:
        features.append(np.array([0.0, 0.0], dtype=np.float32))

    # Opponents bids if known (4): [left_bid/13, left_known, right_bid/13, right_known]
    opp_vec = np.zeros(4, dtype=np.float32)
    for idx, offset in enumerate([1, 3]):
        opp = (player + offset) % 4
        ob = deal.bids[opp]
        if ob >= 0:
            opp_vec[idx * 2] = ob / 13.0
            opp_vec[idx * 2 + 1] = 1.0
    features.append(opp_vec)

    # Scores (2)
    team = deal.team_of(player)
    target = max(game.target_score, 1)
    features.append(np.array([
        game.scores[team] / target,
        game.scores[1 - team] / target
    ], dtype=np.float32))

    # Bags (2)
    features.append(np.array([
        game.bags[team] / 10.0,
        game.bags[1 - team] / 10.0
    ], dtype=np.float32))

    return np.concatenate(features)

BID_FEATURE_DIM = 52 + 4 + 2 + 4 + 2 + 2  # = 66
