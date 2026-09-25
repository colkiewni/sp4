"""
Belief tracker: infer what each player can/can't hold
based on observed play (void inference, card counting).
"""

from __future__ import annotations
from engine import Card, Suit, Rank, DealState, DECK, TrickState
import random
from typing import Optional


class BeliefState:
    """Tracks constraints on hidden hands from one player's perspective."""

    __slots__ = ('observer', 'void', 'known_played')

    def __init__(self, observer: int):
        self.observer = observer
        # void[player] = set of suits they can't have
        self.void: list[set[Suit]] = [set() for _ in range(4)]
        self.known_played: set[Card] = set()

    def observe_trick(self, trick: TrickState):
        """Update beliefs after a completed trick."""
        lead_suit = trick.lead_suit
        leader = trick.lead_player
        for player, card in trick.cards:
            self.known_played.add(card)
            # Non-leader played off-suit → void in lead suit
            if player != leader and card.suit != lead_suit:
                self.void[player].add(lead_suit)

    def observe_partial_trick(self, trick: TrickState):
        """Update beliefs from cards played so far in current trick."""
        if not trick or not trick.cards:
            return
        lead_suit = trick.lead_suit
        leader = trick.lead_player
        for player, card in trick.cards:
            if player != leader and card.suit != lead_suit:
                self.void[player].add(lead_suit)

    def unseen_cards(self, my_hand: set[Card]) -> set[Card]:
        """Cards not in my hand and not yet played."""
        all_cards = set(DECK)
        return all_cards - my_hand - self.known_played

    def possible_cards_for(self, player: int, my_hand: set[Card]) -> set[Card]:
        """Cards that player could hold given constraints."""
        if player == self.observer:
            return set(my_hand)
        unseen = self.unseen_cards(my_hand)
        return {c for c in unseen if c.suit not in self.void[player]}

    def cards_in_current_trick(self, deal: DealState) -> set[Card]:
        """Cards currently on the table in the active trick."""
        if not deal.current_trick:
            return set()
        return {card for _, card in deal.current_trick.cards}

    def sample_consistent_deal(
        self,
        deal: DealState,
        rng: random.Random = None
    ) -> list[set[Card]]:
        """
        Sample a possible assignment of unseen cards to other players,
        consistent with void constraints and hand sizes.
        Returns hands[4] where hands[observer] = actual hand.

        Uses rejection-free constraint-based dealing.
        """
        rng = rng or random.Random()
        me = self.observer
        my_hand = set(deal.hands[me])

        # Cards on the table in current trick (already committed, not in anyone's hand)
        in_trick = self.cards_in_current_trick(deal)

        # Unseen cards not currently on the table
        unseen = self.unseen_cards(my_hand) - in_trick

        others = [p for p in range(4) if p != me]

        # How many cards each other player should hold
        # (not counting cards they've already played to current trick)
        cards_in_trick_by_player = {}
        if deal.current_trick:
            for p, c in deal.current_trick.cards:
                cards_in_trick_by_player[p] = c

        hand_sizes = {}
        for p in others:
            size = len(deal.hands[p])
            # If we're sampling mid-trick, the engine already removed the card
            # from their hand when play_card was called, so deal.hands[p] is correct
            hand_sizes[p] = size

        total_needed = sum(hand_sizes[p] for p in others)
        if total_needed != len(unseen):
            # Fallback: just deal randomly
            # ponytail: this can happen if state is inconsistent, graceful fallback
            cards_list = list(unseen)
            rng.shuffle(cards_list)
            hands = [set() for _ in range(4)]
            hands[me] = my_hand
            idx = 0
            for p in others:
                for _ in range(hand_sizes[p]):
                    if idx < len(cards_list):
                        hands[p].add(cards_list[idx])
                        idx += 1
            return hands

        # Constraint-based sampling with retry
        # ponytail: O(n) retry with ~95% success rate per attempt, max 100 retries
        for _ in range(100):
            result = self._try_sample(unseen, others, hand_sizes, rng)
            if result is not None:
                hands = [set() for _ in range(4)]
                hands[me] = my_hand
                for p, cards in result.items():
                    hands[p] = cards
                return hands

        # Exhausted retries: ignore constraints
        cards_list = list(unseen)
        rng.shuffle(cards_list)
        hands = [set() for _ in range(4)]
        hands[me] = my_hand
        idx = 0
        for p in others:
            hands[p] = set()
            for _ in range(hand_sizes[p]):
                if idx < len(cards_list):
                    hands[p].add(cards_list[idx])
                    idx += 1
        return hands

    def _try_sample(
        self,
        unseen: set[Card],
        others: list[int],
        hand_sizes: dict[int, int],
        rng: random.Random
    ) -> Optional[dict[int, set[Card]]]:
        """Try to assign unseen cards respecting void constraints."""
        # Sort players by most constrained first (fewest possible cards)
        possible = {p: [c for c in unseen if c.suit not in self.void[p]] for p in others}

        # Check feasibility
        for p in others:
            if len(possible[p]) < hand_sizes[p]:
                return None

        assigned: dict[int, set[Card]] = {p: set() for p in others}
        remaining = set(unseen)

        # Deal to most constrained player first
        order = sorted(others, key=lambda p: len(possible[p]))

        for p in order:
            available = [c for c in remaining if c.suit not in self.void[p]]
            need = hand_sizes[p]
            if len(available) < need:
                return None  # retry
            chosen = rng.sample(available, need)
            assigned[p] = set(chosen)
            remaining -= assigned[p]

        return assigned


def make_belief(observer: int, deal: DealState) -> BeliefState:
    """Create a belief state initialized from deal history."""
    bs = BeliefState(observer)
    for trick in deal.trick_history:
        bs.observe_trick(trick)
    if deal.current_trick and deal.current_trick.cards:
        bs.observe_partial_trick(deal.current_trick)
    return bs
