"""
Main entry point: play a game of Spades.
Supports human vs AI, AI vs AI, and benchmarking.
"""

from __future__ import annotations
import random
import sys
import os
from typing import Optional
from engine import (
    Card, Suit, Rank, DealState, GameState, DECK,
    new_game, start_deal, finish_deal, place_bid, play_card,
    legal_plays, legal_bids, RANK_NAMES, SUIT_NAMES
)
from bidding import make_bid
from search import ismcts_choose


# --- Player interfaces ---

class Player:
    """Base class for a player."""
    def choose_bid(self, hand: set[Card], player: int, deal: DealState, game: GameState) -> int:
        raise NotImplementedError

    def choose_card(self, player: int, deal: DealState, game: GameState) -> Card:
        raise NotImplementedError


class RuleBot(Player):
    """Pure rule-based player (fast, no search)."""
    def choose_bid(self, hand, player, deal, game):
        return make_bid(hand, player, deal, game)

    def choose_card(self, player, deal, game):
        moves = legal_plays(deal, player)
        if len(moves) == 1:
            return moves[0]
        return self._heuristic_play(player, deal, game, moves)

    def _heuristic_play(self, player, deal, game, moves):
        trick = deal.current_trick
        team = deal.team_of(player)
        partner = (player + 2) % 4
        my_bid = deal.bids[player]
        team_tricks = deal.team_tricks(team)
        team_bid = deal.team_bid(team)
        need = max(0, team_bid - team_tricks)

        if my_bid == 0:
            return self._play_nil(moves, trick)
        if deal.bids[partner] == 0:
            return self._play_cover_nil(player, moves, trick, deal)
        if not trick.cards:
            return self._lead(player, moves, deal, need)
        return self._follow(player, moves, trick, deal, need)

    def _play_nil(self, moves, trick):
        """Nil bidder: play the lowest legal card to avoid winning."""
        if not trick.cards:
            # Leading as nil: play lowest non-spade, or lowest spade
            non_spades = [c for c in moves if c.suit != Suit.SPADES]
            pool = non_spades if non_spades else moves
            return min(pool, key=lambda c: c.rank)

        lead_suit = trick.lead_suit
        followers = [c for c in moves if c.suit == lead_suit]

        if followers:
            # Follow suit: play highest card that's still under the current winner
            current_best = max(
                (card for _, card in trick.cards),
                key=lambda c: (c.suit == Suit.SPADES, c.rank)
            )
            safe = [c for c in followers if not c.beats(current_best, lead_suit)]
            if safe:
                return max(safe, key=lambda c: c.rank)  # highest safe card
            return min(followers, key=lambda c: c.rank)  # forced to win, play lowest

        # Can't follow: dump highest non-spade, avoid trumping
        non_spades = [c for c in moves if c.suit != Suit.SPADES]
        if non_spades:
            return max(non_spades, key=lambda c: c.rank)
        return min(moves, key=lambda c: c.rank)  # forced spade, play lowest

    def _play_cover_nil(self, player, moves, trick, deal):
        """Partner bid nil: try to win tricks to protect partner."""
        if not trick.cards:
            # Lead high to draw out cards before partner gets stuck
            return max(moves, key=lambda c: (c.suit == Suit.SPADES, c.rank))

        lead_suit = trick.lead_suit
        partner = (player + 2) % 4

        # Check if partner has already played in this trick
        partner_played = None
        for p, card in trick.cards:
            if p == partner:
                partner_played = card
                break

        if partner_played is not None:
            # Partner already played — try to overtake partner if they're winning
            current_winner_player = trick.cards[0][1]
            current_winner = trick.cards[0][1]
            for p, card in trick.cards:
                if card.beats(current_winner, lead_suit):
                    current_winner = card
                    current_winner_player = p

            # If partner is currently winning, we must beat them
            if current_winner_player == partner:
                beaters = [c for c in moves if c.beats(current_winner, lead_suit)]
                if beaters:
                    return min(beaters, key=lambda c: (c.suit == Suit.SPADES, c.rank))

        # Otherwise play normally — try to win
        return self._follow_try_win(moves, trick, lead_suit)

    def _lead(self, player, moves, deal, need):
        """Choose a card to lead."""
        if need <= 0:
            # Contract met — lead low to avoid bags
            non_spades = [c for c in moves if c.suit != Suit.SPADES]
            pool = non_spades if non_spades else moves
            return min(pool, key=lambda c: c.rank)

        # Need tricks: lead winners
        # Try to lead an ace of a side suit
        for card in moves:
            if card.suit != Suit.SPADES and card.rank == Rank.ACE:
                return card

        # Lead high spade if we have them
        spades = [c for c in moves if c.suit == Suit.SPADES]
        if spades and deal.spades_broken:
            high_spade = max(spades, key=lambda c: c.rank)
            if high_spade.rank >= Rank.QUEEN:
                return high_spade

        # Lead from longest side suit (force opponents to trump)
        by_suit = {}
        for c in moves:
            if c.suit != Suit.SPADES:
                by_suit.setdefault(c.suit, []).append(c)
        if by_suit:
            longest = max(by_suit.values(), key=len)
            return max(longest, key=lambda c: c.rank)

        # Only spades left
        return max(moves, key=lambda c: c.rank)

    def _follow(self, player, moves, trick, deal, need):
        """Follow to a trick already in progress."""
        lead_suit = trick.lead_suit

        if need <= 0:
            # Contract met — duck
            return self._duck(moves, trick, lead_suit)

        return self._follow_try_win(moves, trick, lead_suit)

    def _follow_try_win(self, moves, trick, lead_suit):
        """Try to win the trick."""
        # Find current best card
        current_best = trick.cards[0][1]
        for _, card in trick.cards[1:]:
            if card.beats(current_best, lead_suit):
                current_best = card

        # Cards that beat current best
        beaters = [c for c in moves if c.beats(current_best, lead_suit)]
        if beaters:
            # Win with lowest possible winner
            return min(beaters, key=lambda c: (c.suit == Suit.SPADES, c.rank))

        # Can't win: dump lowest
        return min(moves, key=lambda c: (c.suit == Suit.SPADES, c.rank))

    def _duck(self, moves, trick, lead_suit):
        """Try NOT to win the trick (bag avoidance)."""
        current_best = trick.cards[0][1]
        for _, card in trick.cards[1:]:
            if card.beats(current_best, lead_suit):
                current_best = card

        # Play highest card that doesn't win
        losers = [c for c in moves if not c.beats(current_best, lead_suit)]
        if losers:
            return max(losers, key=lambda c: c.rank)
        # Forced to win — play lowest winner
        return min(moves, key=lambda c: (c.suit == Suit.SPADES, c.rank))


class SearchBot(Player):
    """ISMCTS-based player. Uses multiple cores for search."""
    def __init__(self, iterations: int = 3000, num_workers: int = None):
        self.iterations = iterations
        # Default: use half the cores (other half free for opponent's search)
        self.num_workers = num_workers or max(1, (os.cpu_count() or 4) // 2)

    def choose_bid(self, hand, player, deal, game):
        return make_bid(hand, player, deal, game)

    def choose_card(self, player, deal, game):
        return ismcts_choose(
            deal, game, player,
            iterations=self.iterations,
            num_workers=self.num_workers
        )


class HumanPlayer(Player):
    """Interactive human player via terminal."""
    def choose_bid(self, hand, player, deal, game):
        print(f"\nYour hand (Player {player}):")
        _print_hand(hand)
        _print_scores(game)
        while True:
            try:
                bid = int(input(f"Your bid (0=nil, 1-13): "))
                if 0 <= bid <= 13:
                    return bid
                print("Invalid bid.")
            except (ValueError, EOFError):
                print("Enter a number 0-13.")

    def choose_card(self, player, deal, game):
        moves = legal_plays(deal, player)
        print(f"\nYour hand (Player {player}):")
        _print_hand(deal.hands[player])
        if deal.current_trick and deal.current_trick.cards:
            print("Current trick:", [(p, repr(c)) for p, c in deal.current_trick.cards])
        print("Legal plays:")
        for i, card in enumerate(moves):
            print(f"  {i}: {card!r}")
        while True:
            try:
                idx = int(input(f"Choose card (0-{len(moves)-1}): "))
                if 0 <= idx < len(moves):
                    return moves[idx]
                print("Invalid choice.")
            except (ValueError, EOFError):
                print(f"Enter a number 0-{len(moves)-1}.")


# --- Display helpers ---

def _print_hand(hand):
    by_suit = {s: [] for s in Suit}
    for c in hand:
        by_suit[c.suit].append(c)
    for s in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
        cards = sorted(by_suit[s], key=lambda c: c.rank, reverse=True)
        if cards:
            print(f"  {SUIT_NAMES[s]}: {' '.join(repr(c) for c in cards)}")


def _print_scores(game):
    print(f"  Scores: Team 0={game.scores[0]} (bags:{game.bags[0]}) | "
          f"Team 1={game.scores[1]} (bags:{game.bags[1]}) | "
          f"Target: {game.target_score}")


def _print_trick(trick):
    cards_str = ' '.join(f"P{p}:{c!r}" for p, c in trick.cards)
    winner = trick.winner()
    print(f"  Trick: {cards_str} → P{winner} wins")


# --- Game loop ---

def play_game(
    players: list[Player],
    target_score: int = 500,
    verbose: bool = True,
    seed: int = None
) -> GameState:
    """Play a full game of Spades."""
    rng = random.Random(seed)
    game = new_game(target_score=target_score, first_dealer=rng.randint(0, 3))

    while not game.game_over:
        game = start_deal(game, rng)
        deal = game.deal

        if verbose:
            print(f"\n{'='*50}")
            print(f"Deal #{game.deal_number} | Dealer: P{deal.dealer}")
            _print_scores(game)

        # Bidding
        while deal.phase == 'bidding':
            p = deal.current_player
            bid = players[p].choose_bid(set(deal.hands[p]), p, deal, game)
            place_bid(deal, p, bid)
            if verbose:
                label = "NIL" if bid == 0 else str(bid)
                print(f"  P{p} bids {label}")

        if verbose:
            for t in range(2):
                tp = [i for i in range(4) if deal.team_of(i) == t]
                bids = [deal.bids[i] for i in tp]
                print(f"  Team {t}: P{tp[0]} bid {bids[0]}, P{tp[1]} bid {bids[1]} "
                      f"(total: {sum(b for b in bids if b > 0)})")

        # Playing
        while deal.phase == 'playing':
            p = deal.current_player
            card = players[p].choose_card(p, deal, game)
            play_card(deal, p, card)

            if deal.current_trick is None or deal.phase == 'done':
                # Trick just completed (was the last trick)
                last_trick = deal.trick_history[-1]
                if verbose:
                    _print_trick(last_trick)
            elif len(deal.current_trick.cards) == 0:
                # New trick started — previous one just completed
                last_trick = deal.trick_history[-1]
                if verbose:
                    _print_trick(last_trick)

        # Deal done
        if verbose:
            print(f"\n  Results:")
            for t in range(2):
                tp = [i for i in range(4) if deal.team_of(i) == t]
                tricks = [deal.tricks_won[i] for i in tp]
                total = sum(tricks)
                bid = deal.team_bid(t)
                nils = [i for i in tp if deal.bids[i] == 0]
                nil_str = ""
                for n in nils:
                    ok = deal.tricks_won[n] == 0
                    nil_str += f" | P{n} nil {'✓' if ok else '✗'}"
                print(f"  Team {t}: bid {bid}, took {total} "
                      f"(P{tp[0]}:{tricks[0]}, P{tp[1]}:{tricks[1]}){nil_str}")

        game = finish_deal(game)

        if verbose:
            _print_scores(game)

    if verbose:
        print(f"\n{'='*50}")
        print(f"GAME OVER after {game.deal_number} deals")
        print(f"Winner: Team {game.winner}")
        print(f"Final scores: Team 0={game.scores[0]}, Team 1={game.scores[1]}")

    return game


# --- Benchmarking ---

def benchmark(
    num_games: int = 100,
    p0: str = 'rule',
    p1: str = 'rule',
    target_score: int = 500,
    search_iters: int = 3000
):
    """Run many AI vs AI games and report win rates."""
    def make_player(kind):
        if kind == 'rule':
            return RuleBot()
        elif kind == 'search':
            return SearchBot(iterations=search_iters)
        else:
            raise ValueError(f"Unknown player type: {kind}")

    # Team 0 = players 0, 2. Team 1 = players 1, 3.
    players = [make_player(p0), make_player(p1), make_player(p0), make_player(p1)]

    wins = [0, 0]
    total_deals = 0

    for i in range(num_games):
        game = play_game(players, target_score=target_score, verbose=False, seed=i)
        if game.winner is not None:
            wins[game.winner] += 1
        total_deals += game.deal_number

        if (i + 1) % 10 == 0:
            print(f"Games: {i+1}/{num_games} | "
                  f"Team0({p0}): {wins[0]} | Team1({p1}): {wins[1]} | "
                  f"Avg deals/game: {total_deals/(i+1):.1f}")

    print(f"\n{'='*50}")
    print(f"Final: {num_games} games")
    print(f"  Team 0 ({p0}): {wins[0]} wins ({100*wins[0]/num_games:.1f}%)")
    print(f"  Team 1 ({p1}): {wins[1]} wins ({100*wins[1]/num_games:.1f}%)")
    print(f"  Avg deals/game: {total_deals/num_games:.1f}")
    return wins


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spades')
    sub = parser.add_subparsers(dest='cmd')

    # Play interactive game
    p_play = sub.add_parser('play', help='Play interactive game')
    p_play.add_argument('--seat', type=int, default=0, help='Your seat (0-3)')
    p_play.add_argument('--target', type=int, default=500)
    p_play.add_argument('--partner', type=str, default='rule', choices=['rule', 'search'])
    p_play.add_argument('--opponent', type=str, default='rule', choices=['rule', 'search'])
    p_play.add_argument('--search-iters', type=int, default=3000)

    # AI vs AI benchmark
    p_bench = sub.add_parser('bench', help='Benchmark AI vs AI')
    p_bench.add_argument('--games', type=int, default=100)
    p_bench.add_argument('--target', type=int, default=500)
    p_bench.add_argument('--team0', type=str, default='rule', choices=['rule', 'search'])
    p_bench.add_argument('--team1', type=str, default='rule', choices=['rule', 'search'])
    p_bench.add_argument('--search-iters', type=int, default=3000)

    # Watch AI game
    p_watch = sub.add_parser('watch', help='Watch AI vs AI game')
    p_watch.add_argument('--target', type=int, default=500)
    p_watch.add_argument('--team0', type=str, default='rule', choices=['rule', 'search'])
    p_watch.add_argument('--team1', type=str, default='rule', choices=['rule', 'search'])
    p_watch.add_argument('--search-iters', type=int, default=3000)
    p_watch.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.cmd == 'play':
        def _make(kind):
            if kind == 'search':
                return SearchBot(iterations=args.search_iters)
            return RuleBot()

        players = [None] * 4
        players[args.seat] = HumanPlayer()
        partner_seat = (args.seat + 2) % 4
        players[partner_seat] = _make(args.partner)
        for i in range(4):
            if players[i] is None:
                players[i] = _make(args.opponent)
        play_game(players, target_score=args.target)

    elif args.cmd == 'bench':
        benchmark(
            num_games=args.games,
            p0=args.team0,
            p1=args.team1,
            target_score=args.target,
            search_iters=args.search_iters
        )

    elif args.cmd == 'watch':
        def _make(kind):
            if kind == 'search':
                return SearchBot(iterations=args.search_iters)
            return RuleBot()
        players = [_make(args.team0), _make(args.team1),
                    _make(args.team0), _make(args.team1)]
        play_game(players, target_score=args.target, verbose=True, seed=args.seed)

    else:
        parser.print_help()
