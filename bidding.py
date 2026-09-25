"""
Bidding module: rule-based heuristics with score-pressure adjustments.
"""

from __future__ import annotations
from engine import Card, Suit, Rank, DealState, GameState


def count_expected_tricks(hand: set[Card]) -> float:
    """Estimate tricks a hand can take using standard Spades heuristics."""
    tricks = 0.0

    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in hand:
        by_suit[c.suit].append(c)
    for s in by_suit:
        by_suit[s].sort(key=lambda c: c.rank, reverse=True)

    spades = by_suit[Suit.SPADES]
    num_spades = len(spades)

    # Top spades
    for i, card in enumerate(spades):
        if i == 0 and card.rank == Rank.ACE:
            tricks += 1.0
        elif i == 0 and card.rank == Rank.KING:
            tricks += 0.9
        elif i == 0 and card.rank == Rank.QUEEN:
            tricks += 0.6
        elif i == 1 and card.rank >= Rank.KING:
            tricks += 0.9
        elif i == 1 and card.rank >= Rank.QUEEN:
            tricks += 0.7
        elif i == 2 and card.rank >= Rank.QUEEN:
            tricks += 0.5
        elif i >= 3:
            # 4th+ spade: usually a winner
            tricks += 0.7
            break  # don't double-count deeper spades

    # Extra length spades beyond 3
    if num_spades >= 4:
        tricks += (num_spades - 3) * 0.7

    # Side suits
    for suit in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS]:
        cards = by_suit[suit]
        if not cards:
            # Void: potential ruff with each spade beyond what we already counted
            # Already captured in spade length
            continue

        length = len(cards)
        top = cards[0].rank if cards else 0
        second = cards[1].rank if len(cards) >= 2 else 0

        if top == Rank.ACE:
            tricks += 1.0
            if second == Rank.KING:
                tricks += 0.85
            elif second == Rank.QUEEN:
                tricks += 0.4
        elif top == Rank.KING:
            if second == Rank.QUEEN:
                tricks += 0.7
            else:
                tricks += 0.4
        elif top == Rank.QUEEN:
            if second >= Rank.JACK and num_spades >= 4:
                tricks += 0.3

        # Short suit with spades = ruff potential
        if length == 1 and num_spades >= 2:
            tricks += 0.5
        elif length == 0 and num_spades >= 1:
            # Already void — counted via spade length
            pass

    return tricks


def can_nil_safely(hand: set[Card]) -> bool:
    """Check if a nil bid is reasonable."""
    by_suit: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in hand:
        by_suit[c.suit].append(c)
    for s in by_suit:
        by_suit[s].sort(key=lambda c: c.rank)

    spades = by_suit[Suit.SPADES]
    # Any high spade is dangerous for nil
    if any(c.rank >= Rank.QUEEN for c in spades):
        return False
    if len(spades) >= 5:
        return False

    # Check side suits for unavoidable winners
    danger = 0
    for suit in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS]:
        cards = by_suit[suit]
        if not cards:
            continue
        # Ace in a long suit is very dangerous
        if cards[-1].rank == Rank.ACE:
            if len(cards) <= 2:
                danger += 2  # hard to duck
            else:
                danger += 1
        if cards[-1].rank == Rank.KING and len(cards) <= 2:
            danger += 1

    return danger <= 1


def adjust_for_score(
    bid: int,
    hand: set[Card],
    game: GameState,
    team: int,
    partner_bid: int
) -> int:
    """Adjust bid based on score situation."""
    my_score = game.scores[team]
    opp_score = game.scores[1 - team]
    my_bags = game.bags[team]
    target = game.target_score

    # Bag danger: if close to 10 bags, bid conservatively (round down)
    if my_bags >= 7 and bid > 1:
        bid = max(1, bid - 1)

    # Behind on score: bid slightly more aggressively
    if opp_score - my_score > 50 and bid >= 2:
        bid = min(13, bid + 1)

    # Close to winning: be precise
    points_needed = target - my_score
    if 0 < points_needed <= 50:
        # Need exactly this many points, don't overbid
        team_bid_so_far = partner_bid if partner_bid >= 0 else 0
        ideal_team_bid = max(1, (points_needed + 9) // 10)  # ceiling
        remaining = ideal_team_bid - team_bid_so_far
        if remaining > 0:
            bid = min(bid, max(1, remaining))

    return max(1, min(13, bid))


def make_bid(
    hand: set[Card],
    player: int,
    deal: DealState,
    game: GameState
) -> int:
    """Produce a bid for the given player."""
    team = deal.team_of(player)
    partner = (player + 2) % 4
    partner_bid = deal.bids[partner]  # -1 if not yet bid

    expected = count_expected_tricks(hand)

    # Consider nil
    if expected <= 0.8 and can_nil_safely(hand):
        return 0

    bid = max(1, round(expected))

    # Adjust for game state
    if partner_bid >= 0:
        bid = adjust_for_score(bid, hand, game, team, partner_bid)
    else:
        bid = adjust_for_score(bid, hand, game, team, -1)

    # Don't let team bid exceed 13 (unless aggressive)
    if partner_bid > 0:
        if bid + partner_bid > 13:
            bid = max(1, 13 - partner_bid)

    return max(1, min(13, bid)) if bid != 0 else 0
