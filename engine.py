"""Spades game engine. Complete rules, state management, scoring."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import random

# --- Cards ---

class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

class Rank(IntEnum):
    TWO = 2; THREE = 3; FOUR = 4; FIVE = 5; SIX = 6; SEVEN = 7
    EIGHT = 8; NINE = 9; TEN = 10; JACK = 11; QUEEN = 12; KING = 13; ACE = 14

SUIT_NAMES = {Suit.CLUBS: '♣', Suit.DIAMONDS: '♦', Suit.HEARTS: '♥', Suit.SPADES: '♠'}
RANK_NAMES = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}

@dataclass(frozen=True, slots=True, order=True)
class Card:
    suit: Suit
    rank: Rank

    def __repr__(self):
        return f"{RANK_NAMES[self.rank]}{SUIT_NAMES[self.suit]}"

    def beats(self, other: Card, lead_suit: Suit) -> bool:
        """Does self beat other given lead_suit?"""
        s_trump = self.suit == Suit.SPADES
        o_trump = other.suit == Suit.SPADES
        if s_trump and not o_trump:
            return True
        if o_trump and not s_trump:
            return False
        if s_trump and o_trump:
            return self.rank > other.rank
        # Neither is trump
        s_follow = self.suit == lead_suit
        o_follow = other.suit == lead_suit
        if s_follow and not o_follow:
            return True
        if o_follow and not s_follow:
            return False
        return self.rank > other.rank

DECK = [Card(s, r) for s in Suit for r in Rank]
assert len(DECK) == 52


# --- State ---

@dataclass
class TrickState:
    lead_player: int
    cards: list  # list of (player, Card), up to 4
    lead_suit: Optional[Suit] = None

    def add(self, player: int, card: Card):
        if not self.cards:
            self.lead_suit = card.suit
        self.cards.append((player, card))

    def is_complete(self) -> bool:
        return len(self.cards) == 4

    def winner(self) -> int:
        assert self.is_complete()
        best_player, best_card = self.cards[0]
        for player, card in self.cards[1:]:
            if card.beats(best_card, self.lead_suit):
                best_player, best_card = player, card
        return best_player


@dataclass
class DealState:
    hands: list[set[Card]]          # hands[player] = set of Cards
    bids: list[int]                 # bids[player], -1 = not yet bid
    tricks_won: list[int]           # tricks_won[player]
    dealer: int
    current_trick: Optional[TrickState] = None
    trick_number: int = 0           # 0-based, 0..12
    spades_broken: bool = False
    cards_played: set[Card] = field(default_factory=set)
    trick_history: list[TrickState] = field(default_factory=list)
    current_player: int = -1        # who plays next
    phase: str = 'bidding'          # 'bidding' | 'playing' | 'done'

    @property
    def turn_order(self) -> list[int]:
        """Bidding and first lead: start from dealer."""
        return [(self.dealer + i) % 4 for i in range(4)]

    def next_bidder(self) -> Optional[int]:
        for p in self.turn_order:
            if self.bids[p] == -1:
                return p
        return None

    def team_of(self, player: int) -> int:
        return player % 2  # 0,2 = team 0; 1,3 = team 1

    def team_bid(self, team: int) -> int:
        """Combined non-nil bid for a team."""
        total = 0
        for p in range(4):
            if self.team_of(p) == team and self.bids[p] > 0:
                total += self.bids[p]
        return total

    def team_tricks(self, team: int) -> int:
        return sum(self.tricks_won[p] for p in range(4) if self.team_of(p) == team)


@dataclass
class GameState:
    scores: list[int] = field(default_factory=lambda: [0, 0])
    bags: list[int] = field(default_factory=lambda: [0, 0])
    dealer: int = 0
    target_score: int = 500
    deal_number: int = 0
    deal: Optional[DealState] = None
    game_over: bool = False
    winner: Optional[int] = None  # winning team


# --- Dealing ---

def deal_cards(dealer: int, rng: random.Random = None) -> DealState:
    rng = rng or random.Random()
    deck = list(DECK)
    rng.shuffle(deck)
    hands = [set() for _ in range(4)]
    for i, card in enumerate(deck):
        hands[i % 4].add(card)
    ds = DealState(
        hands=hands,
        bids=[-1, -1, -1, -1],
        tricks_won=[0, 0, 0, 0],
        dealer=dealer,
    )
    ds.current_player = ds.turn_order[0]
    return ds


# --- Legal moves ---

def legal_bids(deal: DealState, player: int) -> list[int]:
    """Legal bids for a player: 0 (nil) through 13."""
    # ponytail: no blind nil per spec
    return list(range(0, 14))


def legal_plays(deal: DealState, player: int) -> list[Card]:
    """Legal cards for player to play in current trick."""
    hand = deal.hands[player]
    if not hand:
        return []

    trick = deal.current_trick
    lead_suit = trick.lead_suit if trick and trick.cards else None

    if lead_suit is not None:
        # Must follow suit if possible
        followers = [c for c in hand if c.suit == lead_suit]
        if followers:
            return sorted(followers)
        # Can't follow: play anything
        return sorted(hand)

    # Leading
    if not deal.spades_broken:
        non_spades = [c for c in hand if c.suit != Suit.SPADES]
        if non_spades:
            return sorted(non_spades)
        # Only spades: allowed to lead them
    return sorted(hand)


# --- Actions ---

def place_bid(deal: DealState, player: int, bid: int) -> DealState:
    assert deal.phase == 'bidding'
    assert deal.bids[player] == -1
    assert 0 <= bid <= 13
    deal.bids[player] = bid
    nxt = deal.next_bidder()
    if nxt is None:
        # Bidding complete, start playing
        deal.phase = 'playing'
        deal.current_player = deal.dealer  # dealer leads first trick
        deal.current_trick = TrickState(lead_player=deal.dealer, cards=[])
    else:
        deal.current_player = nxt
    return deal


def play_card(deal: DealState, player: int, card: Card) -> DealState:
    assert deal.phase == 'playing'
    assert player == deal.current_player
    assert card in deal.hands[player]
    assert card in legal_plays(deal, player)

    deal.hands[player].remove(card)
    deal.cards_played.add(card)
    deal.current_trick.add(player, card)

    # Track spades broken
    if card.suit == Suit.SPADES and not deal.spades_broken:
        if deal.current_trick.lead_suit != Suit.SPADES:
            deal.spades_broken = True
        # If leading spades (only spades in hand), also counts as broken
        elif len(deal.current_trick.cards) == 1:
            deal.spades_broken = True

    if deal.current_trick.is_complete():
        winner = deal.current_trick.winner()
        deal.tricks_won[winner] += 1
        deal.trick_history.append(deal.current_trick)
        deal.trick_number += 1

        if deal.trick_number == 13:
            deal.phase = 'done'
            deal.current_player = -1
        else:
            deal.current_trick = TrickState(lead_player=winner, cards=[])
            deal.current_player = winner
    else:
        # Next player in clockwise order
        deal.current_player = (player + 1) % 4

    return deal


# --- Scoring ---

def score_deal(deal: DealState, game: GameState) -> tuple[list[int], list[int]]:
    """
    Returns (score_deltas[2], new_bags[2]).
    score_deltas includes bag penalties.
    """
    assert deal.phase == 'done'

    score_deltas = [0, 0]
    new_bags = list(game.bags)

    for team in range(2):
        players = [p for p in range(4) if deal.team_of(p) == team]
        nils = [p for p in players if deal.bids[p] == 0]
        non_nils = [p for p in players if deal.bids[p] > 0]

        # --- Nil scoring ---
        for p in nils:
            if deal.tricks_won[p] == 0:
                score_deltas[team] += 100
            else:
                score_deltas[team] -= 100

        # --- Normal contract scoring ---
        team_bid = sum(deal.bids[p] for p in non_nils)
        # Tricks that count toward contract: all team tricks
        # (a nil player's tricks count against nil but the tricks
        #  still count toward the team's total for the non-nil partner)
        team_tricks = sum(deal.tricks_won[p] for p in players)
        # For contract purposes, only non-nil tricks matter?
        # Per spec: "A failed nil still incurs the normal partnership scoring
        #  based on the partner's bid."
        # The team's tricks (including nil player's) count toward partner's bid.

        if team_bid == 0:
            # Both players bid nil (double nil) — no normal contract
            pass
        else:
            # Boston check: team bids 13, takes all 13
            # "Not possible when opponents have double nil"
            opp_team = 1 - team
            opp_players = [p for p in range(4) if deal.team_of(p) == opp_team]
            opp_double_nil = all(deal.bids[p] == 0 for p in opp_players)

            if team_bid == 13 and team_tricks == 13 and not opp_double_nil:
                score_deltas[team] += 200
            elif team_tricks >= team_bid:
                score_deltas[team] += 10 * team_bid
                overtricks = team_tricks - team_bid
                new_bags[team] += overtricks
                score_deltas[team] += overtricks  # +1 per bag

                # Bag penalty: every 10 cumulative bags = -100
                if new_bags[team] >= 10:
                    bag_penalties = new_bags[team] // 10
                    score_deltas[team] -= 100 * bag_penalties
                    new_bags[team] = new_bags[team] % 10
            else:
                # Failed contract
                score_deltas[team] -= 10 * team_bid

    return score_deltas, new_bags


def apply_deal_score(game: GameState, deal: DealState) -> GameState:
    """Apply deal results to game state. Check win/loss conditions."""
    deltas, new_bags = score_deal(deal, game)
    game.scores[0] += deltas[0]
    game.scores[1] += deltas[1]
    game.bags = new_bags

    # Check -200 cutoff
    for team in range(2):
        if game.scores[team] <= -200:
            game.game_over = True
            # Other team wins (or team with higher score)
            if game.scores[0] != game.scores[1]:
                game.winner = 0 if game.scores[0] > game.scores[1] else 1
            else:
                game.winner = 1 - team  # other team wins on cutoff
            return game

    # Check target score
    both_reached = (game.scores[0] >= game.target_score and
                    game.scores[1] >= game.target_score)
    if both_reached:
        game.game_over = True
        game.winner = 0 if game.scores[0] >= game.scores[1] else 1
    elif game.scores[0] >= game.target_score:
        game.game_over = True
        game.winner = 0
    elif game.scores[1] >= game.target_score:
        game.game_over = True
        game.winner = 1

    return game


def new_game(target_score: int = 500, first_dealer: int = 0) -> GameState:
    return GameState(target_score=target_score, dealer=first_dealer)


def start_deal(game: GameState, rng: random.Random = None) -> GameState:
    game.deal = deal_cards(game.dealer, rng)
    game.deal_number += 1
    return game


def finish_deal(game: GameState) -> GameState:
    assert game.deal and game.deal.phase == 'done'
    game = apply_deal_score(game, game.deal)
    game.dealer = (game.dealer + 1) % 4
    game.deal = None
    return game
