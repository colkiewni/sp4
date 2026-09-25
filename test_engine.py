"""
Engine correctness tests. Run with: python test_engine.py
Covers rules, scoring, edge cases.
"""

from engine import *
from bidding import count_expected_tricks, can_nil_safely, make_bid
from belief import BeliefState, make_belief
from features import encode_state, encode_hand_for_bidding, FEATURE_DIM, BID_FEATURE_DIM
import random
import sys

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def test_card_basics():
    print("test_card_basics")
    c1 = Card(Suit.SPADES, Rank.ACE)
    c2 = Card(Suit.HEARTS, Rank.ACE)
    c3 = Card(Suit.SPADES, Rank.TWO)
    c4 = Card(Suit.HEARTS, Rank.KING)

    check(c1.beats(c2, Suit.HEARTS), "Ace of spades beats ace of hearts")
    check(c3.beats(c2, Suit.HEARTS), "2 of spades beats ace of hearts (trump)")
    check(not c2.beats(c3, Suit.HEARTS), "Ace of hearts doesn't beat 2 of spades")
    check(c1.beats(c3, Suit.SPADES), "Ace of spades beats 2 of spades")
    check(not c3.beats(c1, Suit.SPADES), "2 of spades doesn't beat ace of spades")
    check(c2.beats(c4, Suit.HEARTS), "Ace of hearts beats king of hearts")

    c5 = Card(Suit.DIAMONDS, Rank.ACE)
    check(not c5.beats(c4, Suit.HEARTS), "Ace of diamonds doesn't beat king of hearts when hearts led")
    check(c4.beats(c5, Suit.HEARTS), "King of hearts beats ace of diamonds when hearts led")

    check(len(DECK) == 52, "Deck has 52 cards")
    check(len(set(DECK)) == 52, "All cards unique")


def test_deal():
    print("test_deal")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    check(len(ds.hands[0]) == 13, "Player 0 has 13 cards")
    check(len(ds.hands[1]) == 13, "Player 1 has 13 cards")
    check(len(ds.hands[2]) == 13, "Player 2 has 13 cards")
    check(len(ds.hands[3]) == 13, "Player 3 has 13 cards")

    all_cards = set()
    for h in ds.hands:
        all_cards |= h
    check(len(all_cards) == 52, "All 52 cards dealt")
    check(ds.dealer == 0, "Dealer is 0")
    check(ds.phase == 'bidding', "Phase is bidding")
    check(ds.current_player == 0, "First bidder is dealer")


def test_bidding():
    print("test_bidding")
    rng = random.Random(42)
    ds = deal_cards(0, rng)

    for p in range(4):
        place_bid(ds, p, 3)

    check(ds.phase == 'playing', "After 4 bids, phase is playing")
    check(ds.bids == [3, 3, 3, 3], "All bids recorded")
    check(ds.current_player == 0, "Dealer leads first trick")


def test_trick_resolution():
    print("test_trick_resolution")
    ts = TrickState(lead_player=0, cards=[])
    ts.add(0, Card(Suit.HEARTS, Rank.KING))
    ts.add(1, Card(Suit.HEARTS, Rank.ACE))
    ts.add(2, Card(Suit.HEARTS, Rank.QUEEN))
    ts.add(3, Card(Suit.HEARTS, Rank.JACK))
    check(ts.winner() == 1, "Ace of hearts wins hearts trick")

    ts2 = TrickState(lead_player=0, cards=[])
    ts2.add(0, Card(Suit.HEARTS, Rank.ACE))
    ts2.add(1, Card(Suit.SPADES, Rank.TWO))
    ts2.add(2, Card(Suit.HEARTS, Rank.KING))
    ts2.add(3, Card(Suit.DIAMONDS, Rank.ACE))
    check(ts2.winner() == 1, "2 of spades trumps ace of hearts")

    ts3 = TrickState(lead_player=0, cards=[])
    ts3.add(0, Card(Suit.HEARTS, Rank.ACE))
    ts3.add(1, Card(Suit.SPADES, Rank.TWO))
    ts3.add(2, Card(Suit.SPADES, Rank.KING))
    ts3.add(3, Card(Suit.DIAMONDS, Rank.ACE))
    check(ts3.winner() == 2, "King of spades beats 2 of spades")


def test_legal_plays():
    print("test_legal_plays")
    rng = random.Random(99)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    moves = legal_plays(ds, 0)
    has_non_spade = any(c.suit != Suit.SPADES for c in ds.hands[0])
    if has_non_spade:
        spade_in_moves = any(c.suit == Suit.SPADES for c in moves)
        check(not spade_in_moves, "Can't lead spades when unbroken and have non-spades")
    else:
        check(len(moves) > 0, "Can lead spades if only spades")

    for c in moves:
        check(c in ds.hands[0], f"Legal move {c!r} is in hand")


def test_follow_suit():
    print("test_follow_suit")
    rng = random.Random(200)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    # Play first card
    moves0 = legal_plays(ds, 0)
    lead = moves0[0]
    play_card(ds, 0, lead)
    lead_suit = lead.suit

    # Next player must follow suit if able
    p1 = ds.current_player
    moves1 = legal_plays(ds, p1)
    has_lead_suit = any(c.suit == lead_suit for c in ds.hands[p1])
    if has_lead_suit:
        check(all(c.suit == lead_suit for c in moves1),
              f"P{p1} must follow {SUIT_NAMES[lead_suit]}")
    else:
        # Can play anything
        check(len(moves1) == len(ds.hands[p1]),
              f"P{p1} void in {SUIT_NAMES[lead_suit]}, can play anything")


def test_spades_broken():
    print("test_spades_broken")
    rng = random.Random(123)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    check(not ds.spades_broken, "Spades not broken initially")

    lead_card = None
    for c in sorted(ds.hands[0]):
        if c.suit != Suit.SPADES:
            lead_card = c
            break
    if lead_card is None:
        print("  (skipped — player 0 has all spades)")
        return

    play_card(ds, 0, lead_card)

    for p_idx in range(1, 4):
        p = (0 + p_idx) % 4
        moves = legal_plays(ds, p)
        card = moves[0]
        play_card(ds, p, card)

    spade_played = any(
        card.suit == Suit.SPADES and card.suit != ds.trick_history[-1].lead_suit
        for _, card in ds.trick_history[-1].cards
    )
    if spade_played:
        check(ds.spades_broken, "Spades broken after trump played")


def test_full_deal():
    print("test_full_deal")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    while ds.phase == 'playing':
        p = ds.current_player
        moves = legal_plays(ds, p)
        check(len(moves) > 0, f"Player {p} has legal moves on trick {ds.trick_number}")
        card = moves[0]
        play_card(ds, p, card)

    check(ds.phase == 'done', "Deal completed")
    check(ds.trick_number == 13, "13 tricks played")
    total_tricks = sum(ds.tricks_won)
    check(total_tricks == 13, f"Total tricks won = 13 (got {total_tricks})")
    check(len(ds.trick_history) == 13, "13 tricks in history")

    for p in range(4):
        check(len(ds.hands[p]) == 0, f"Player {p} hand empty after deal")


def test_scoring_basic():
    print("test_scoring_basic")
    game = new_game(target_score=500)
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    ds.bids = [3, 3, 3, 3]
    ds.tricks_won = [4, 3, 3, 3]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: bid 6, took 7 → +60 +1 bag = +61
    check(deltas[0] == 61, f"Team 0 score delta = 61 (got {deltas[0]})")
    check(new_bags[0] == 1, f"Team 0 bags = 1 (got {new_bags[0]})")
    # Team 1: bid 6, took 6 → +60
    check(deltas[1] == 60, f"Team 1 score delta = 60 (got {deltas[1]})")
    check(new_bags[1] == 0, f"Team 1 bags = 0 (got {new_bags[1]})")


def test_scoring_failed():
    print("test_scoring_failed")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [5, 4, 3, 2]
    ds.tricks_won = [2, 4, 2, 5]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: P0 bid 5, P2 bid 3 → team bid 8, took 4 → failed → -80
    check(deltas[0] == -80, f"Team 0 failed contract = -80 (got {deltas[0]})")
    # Team 1: P1 bid 4, P3 bid 2 → team bid 6, took 9 → +60 +3 bags = +63
    check(deltas[1] == 63, f"Team 1 score delta = 63 (got {deltas[1]})")
    check(new_bags[1] == 3, f"Team 1 bags = 3 (got {new_bags[1]})")


def test_scoring_nil_success():
    print("test_scoring_nil_success")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [0, 3, 5, 3]
    ds.tricks_won = [0, 4, 6, 3]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: P0 nil success (+100), P2 bid 5, team took 6 → +50 +1 bag
    # Total: +100 +50 +1 = +151
    check(deltas[0] == 151, f"Team 0 nil success = 151 (got {deltas[0]})")
    check(new_bags[0] == 1, f"Team 0 bags = 1 (got {new_bags[0]})")


def test_scoring_nil_failed():
    print("test_scoring_nil_failed")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [0, 3, 5, 3]
    ds.tricks_won = [2, 4, 4, 3]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: P0 nil failed (-100), P2 bid 5, team took 6 → +50 +1 bag
    # Total: -100 +50 +1 = -49
    check(deltas[0] == -49, f"Team 0 nil failed = -49 (got {deltas[0]})")


def test_scoring_bags_penalty():
    print("test_scoring_bags_penalty")
    game = new_game(target_score=500)
    game.bags = [8, 0]  # Team 0 already has 8 bags

    ds = deal_cards(0, random.Random(1))
    ds.bids = [3, 3, 3, 3]
    ds.tricks_won = [5, 3, 3, 2]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: bid 6, took 8 → +60 +2 bags = +62
    # But 8+2 = 10 bags → -100 penalty, bags reset to 0
    # Net: +62 -100 = -38
    check(deltas[0] == -38, f"Team 0 bag penalty = -38 (got {deltas[0]})")
    check(new_bags[0] == 0, f"Team 0 bags reset to 0 (got {new_bags[0]})")


def test_scoring_boston():
    print("test_scoring_boston")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [7, 3, 6, 3]
    ds.tricks_won = [7, 0, 6, 0]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: bid 13, took 13 → Boston +200
    check(deltas[0] == 200, f"Team 0 boston = 200 (got {deltas[0]})")


def test_scoring_boston_blocked_by_double_nil():
    print("test_scoring_boston_blocked_by_double_nil")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [7, 0, 6, 0]
    ds.tricks_won = [7, 0, 6, 0]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 1 has double nil — boston not possible for team 0
    # Team 0: bid 13, took 13 → normal scoring: +130
    check(deltas[0] == 130, f"Team 0 no boston (double nil blocks) = 130 (got {deltas[0]})")
    # Team 1: double nil, both succeeded → +200
    check(deltas[1] == 200, f"Team 1 double nil success = 200 (got {deltas[1]})")


def test_scoring_double_nil_one_fails():
    print("test_scoring_double_nil_one_fails")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [5, 0, 5, 0]
    ds.tricks_won = [5, 2, 5, 1]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: bid 10, took 10 → +100
    check(deltas[0] == 100, f"Team 0 = 100 (got {deltas[0]})")
    # Team 1: P1 nil failed (-100), P3 nil failed (-100), no normal contract (both nil)
    # Total: -200
    check(deltas[1] == -200, f"Team 1 double nil both failed = -200 (got {deltas[1]})")


def test_game_win_condition():
    print("test_game_win_condition")
    game = new_game(target_score=100)
    game.scores = [90, 50]

    ds = deal_cards(0, random.Random(1))
    ds.bids = [3, 3, 3, 3]
    ds.tricks_won = [4, 3, 3, 3]
    ds.phase = 'done'

    game = apply_deal_score(game, ds)
    # Team 0: 90 + 61 = 151 ≥ 100 → wins
    check(game.game_over, "Game is over")
    check(game.winner == 0, f"Team 0 wins (got winner={game.winner})")


def test_game_both_reach_target():
    print("test_game_both_reach_target")
    game = new_game(target_score=100)
    game.scores = [95, 95]

    ds = deal_cards(0, random.Random(1))
    ds.bids = [3, 4, 3, 3]
    ds.tricks_won = [3, 4, 3, 3]
    ds.phase = 'done'

    game = apply_deal_score(game, ds)
    # Team 0: 95 + 60 = 155
    # Team 1: 95 + 70 = 165
    # Both ≥ 100, team 1 higher → team 1 wins
    check(game.game_over, "Game is over")
    check(game.winner == 1, f"Team 1 wins with higher score (got winner={game.winner})")


def test_game_negative_cutoff():
    print("test_game_negative_cutoff")
    game = new_game(target_score=500)
    game.scores = [-150, 100]

    ds = deal_cards(0, random.Random(1))
    ds.bids = [5, 3, 5, 3]
    ds.tricks_won = [2, 4, 2, 5]
    ds.phase = 'done'

    game = apply_deal_score(game, ds)
    # Team 0: -150 + (-100) = -250 ≤ -200 → game ends
    check(game.game_over, "Game ends at -200")
    check(game.winner == 1, f"Team 1 wins (got winner={game.winner})")


def test_full_game_completes():
    print("test_full_game_completes")
    from play import RuleBot, play_game
    game = play_game(
        [RuleBot(), RuleBot(), RuleBot(), RuleBot()],
        target_score=100,
        verbose=False,
        seed=42
    )
    check(game.game_over, "Game completed")
    check(game.winner in (0, 1), f"Valid winner (got {game.winner})")
    check(game.deal_number >= 1, f"At least 1 deal played (got {game.deal_number})")
    print(f"  Game finished in {game.deal_number} deals, "
          f"score: {game.scores}, winner: team {game.winner}")


def test_multiple_games_consistency():
    print("test_multiple_games_consistency")
    from play import RuleBot, play_game
    wins = [0, 0]
    for seed in range(20):
        game = play_game(
            [RuleBot(), RuleBot(), RuleBot(), RuleBot()],
            target_score=200,
            verbose=False,
            seed=seed
        )
        check(game.game_over, f"Game {seed} completed")
        check(game.winner in (0, 1), f"Game {seed} has valid winner")
        # Verify winning condition
        if game.scores[0] <= -200 or game.scores[1] <= -200:
            # Negative cutoff
            pass
        else:
            check(
                game.scores[game.winner] >= game.target_score,
                f"Game {seed}: winner reached target "
                f"(scores={game.scores}, target={game.target_score})"
            )
        wins[game.winner] += 1

    print(f"  20 games: Team 0 won {wins[0]}, Team 1 won {wins[1]}")
    # With symmetric bots, should be roughly even
    check(wins[0] + wins[1] == 20, "All 20 games had a winner")


def test_belief_void_tracking():
    print("test_belief_void_tracking")
    bs = BeliefState(observer=0)

    # Simulate: player 1 didn't follow hearts
    trick = TrickState(lead_player=0, cards=[])
    trick.add(0, Card(Suit.HEARTS, Rank.ACE))
    trick.add(1, Card(Suit.CLUBS, Rank.THREE))  # didn't follow hearts
    trick.add(2, Card(Suit.HEARTS, Rank.KING))
    trick.add(3, Card(Suit.HEARTS, Rank.QUEEN))

    bs.observe_trick(trick)

    check(Suit.HEARTS in bs.void[1], "Player 1 void in hearts")
    check(Suit.HEARTS not in bs.void[0], "Player 0 not void in hearts")
    check(Suit.HEARTS not in bs.void[2], "Player 2 not void in hearts")
    check(Suit.HEARTS not in bs.void[3], "Player 3 not void in hearts")


def test_belief_sampling():
    print("test_belief_sampling")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    belief = make_belief(0, ds)
    sampled = belief.sample_consistent_deal(ds, random.Random(99))

    # My hand should be unchanged
    check(sampled[0] == ds.hands[0], "Observer hand unchanged")

    # Total cards should be correct
    total = sum(len(h) for h in sampled)
    check(total == 52, f"Total cards = 52 (got {total})")

    # No overlaps
    all_cards = set()
    for h in sampled:
        check(len(all_cards & h) == 0, "No card overlap between hands")
        all_cards |= h
    check(len(all_cards) == 52, "All 52 cards accounted for")


def test_belief_sampling_with_voids():
    print("test_belief_sampling_with_voids")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    belief = make_belief(0, ds)
    # Manually mark player 1 void in hearts
    belief.void[1].add(Suit.HEARTS)

    for trial in range(10):
        sampled = belief.sample_consistent_deal(ds, random.Random(trial))
        hearts_in_p1 = [c for c in sampled[1] if c.suit == Suit.HEARTS]
        check(len(hearts_in_p1) == 0,
              f"Trial {trial}: P1 has no hearts (got {len(hearts_in_p1)})")


def test_feature_encoding():
    print("test_feature_encoding")
    rng = random.Random(42)
    game = new_game(target_score=500)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    # Encode for player 0
    feat = encode_state(ds, game, 0)
    check(feat.shape == (FEATURE_DIM,), f"Feature dim = {FEATURE_DIM} (got {feat.shape})")
    check(feat.dtype.name == 'float32', "Feature dtype is float32")

    # Hand encoding: should have exactly 13 ones in first 52 dims
    hand_sum = feat[:52].sum()
    check(abs(hand_sum - 13.0) < 0.01, f"Hand has 13 cards encoded (got {hand_sum})")

    # No cards played yet
    played_sum = feat[52:104].sum()
    check(abs(played_sum - 0.0) < 0.01, f"No cards played yet (got {played_sum})")


def test_bid_feature_encoding():
    print("test_bid_feature_encoding")
    rng = random.Random(42)
    game = new_game(target_score=500)
    ds = deal_cards(0, rng)

    feat = encode_hand_for_bidding(ds.hands[0], 0, ds, game)
    check(feat.shape == (BID_FEATURE_DIM,), f"Bid feature dim = {BID_FEATURE_DIM} (got {feat.shape})")

    hand_sum = feat[:52].sum()
    check(abs(hand_sum - 13.0) < 0.01, f"Hand has 13 cards (got {hand_sum})")


def test_bidding_heuristics():
    print("test_bidding_heuristics")

    # Strong hand: A♠ K♠ Q♠ J♠ + A♥ K♥ + A♦ K♦ + A♣ + low cards
    strong_hand = {
        Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.KING),
        Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.JACK),
        Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.THREE), Card(Suit.CLUBS, Rank.FOUR),
        Card(Suit.DIAMONDS, Rank.TWO), Card(Suit.HEARTS, Rank.TWO),
    }
    expected = count_expected_tricks(strong_hand)
    check(expected >= 8, f"Strong hand expected ≥ 8 tricks (got {expected:.1f})")

    # Weak hand: all low cards
    weak_hand = {
        Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE]
    } | {
        Card(Suit.DIAMONDS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR]
    } | {
        Card(Suit.CLUBS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR]
    } | {
        Card(Suit.SPADES, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR]
    }
    expected_weak = count_expected_tricks(weak_hand)
    check(expected_weak <= 2, f"Weak hand expected ≤ 2 tricks (got {expected_weak:.1f})")

    # Nil candidate
    check(can_nil_safely(weak_hand), "Weak hand can nil safely")
    check(not can_nil_safely(strong_hand), "Strong hand cannot nil safely")


def test_only_spades_can_lead():
    print("test_only_spades_can_lead")
    # Construct a hand with only spades
    rng = random.Random(42)
    ds = deal_cards(0, rng)

    # Force player 0 to have only spades
    all_spades = [c for c in DECK if c.suit == Suit.SPADES]
    # Give player 0 all 13 spades
    ds.hands[0] = set(all_spades)
    remaining = [c for c in DECK if c.suit != Suit.SPADES]
    rng.shuffle(remaining)
    for i, p in enumerate([1, 2, 3]):
        ds.hands[p] = set(remaining[i*13:(i+1)*13])

    # Verify we have exactly 39 non-spades
    check(len(remaining) == 39, f"39 non-spade cards (got {len(remaining)})")

    for p in range(4):
        place_bid(ds, p, 3)

    # Player 0 leads — only has spades, should be allowed
    moves = legal_plays(ds, 0)
    check(len(moves) == 13, f"All 13 spades are legal leads (got {len(moves)})")
    check(all(c.suit == Suit.SPADES for c in moves), "All legal moves are spades")


def test_deal_rotation():
    print("test_deal_rotation")
    game = new_game(target_score=500)
    check(game.dealer == 0, "Initial dealer is 0")

    from play import RuleBot, play_game
    # Play a game and check dealer rotates
    rng = random.Random(42)
    game = start_deal(game, rng)
    deal = game.deal
    check(deal.dealer == 0, "First deal dealer is 0")

    # Complete the deal
    for p in range(4):
        place_bid(deal, p, 3)
    while deal.phase == 'playing':
        p = deal.current_player
        moves = legal_plays(deal, p)
        play_card(deal, p, moves[0])

    game = finish_deal(game)
    check(game.dealer == 1, f"Dealer rotated to 1 (got {game.dealer})")


def test_cards_played_tracking():
    print("test_cards_played_tracking")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    cards_seen = set()
    while ds.phase == 'playing':
        p = ds.current_player
        moves = legal_plays(ds, p)
        card = moves[0]
        play_card(ds, p, card)
        cards_seen.add(card)
        check(card in ds.cards_played, f"{card!r} tracked in cards_played")

    check(len(ds.cards_played) == 52, f"All 52 cards tracked (got {len(ds.cards_played)})")
    check(ds.cards_played == cards_seen, "cards_played matches what we saw")


def test_trick_winner_leads_next():
    print("test_trick_winner_leads_next")
    rng = random.Random(42)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    prev_winner = None
    while ds.phase == 'playing':
        p = ds.current_player

        # If this is the start of a new trick and we know who won last
        if ds.current_trick and len(ds.current_trick.cards) == 0 and prev_winner is not None:
            check(p == prev_winner,
                  f"Trick winner P{prev_winner} leads next (got P{p})")

        moves = legal_plays(ds, p)
        play_card(ds, p, moves[0])

        # Check if trick just completed
        if ds.current_trick and len(ds.current_trick.cards) == 0 and ds.trick_history:
            prev_winner = ds.trick_history[-1].winner()
        elif ds.phase == 'done' and ds.trick_history:
            prev_winner = ds.trick_history[-1].winner()


def test_no_duplicate_cards_in_play():
    print("test_no_duplicate_cards_in_play")
    for seed in range(10):
        rng = random.Random(seed)
        ds = deal_cards(0, rng)
        for p in range(4):
            place_bid(ds, p, 3)

        all_played = []
        while ds.phase == 'playing':
            p = ds.current_player
            moves = legal_plays(ds, p)
            card = moves[0]
            check(card not in all_played,
                  f"Seed {seed}: {card!r} not played twice")
            all_played.append(card)
            play_card(ds, p, card)

        check(len(all_played) == 52,
              f"Seed {seed}: exactly 52 cards played (got {len(all_played)})")


def test_follow_suit_enforced():
    print("test_follow_suit_enforced")
    for seed in range(20):
        rng = random.Random(seed)
        ds = deal_cards(0, rng)
        for p in range(4):
            place_bid(ds, p, 3)

        while ds.phase == 'playing':
            p = ds.current_player
            moves = legal_plays(ds, p)
            hand = ds.hands[p]

            if ds.current_trick and ds.current_trick.cards:
                lead_suit = ds.current_trick.lead_suit
                has_suit = any(c.suit == lead_suit for c in hand)
                if has_suit:
                    check(all(c.suit == lead_suit for c in moves),
                          f"Seed {seed} P{p}: must follow {SUIT_NAMES[lead_suit]}")

            play_card(ds, p, moves[0])


def test_scoring_edge_exact_bags():
    print("test_scoring_edge_exact_bags")
    # Exactly 10 bags accumulated
    game = new_game(target_score=500)
    game.bags = [7, 0]

    ds = deal_cards(0, random.Random(1))
    ds.bids = [3, 3, 3, 3]
    ds.tricks_won = [5, 3, 3, 2]  # team 0 took 8, bid 6, +2 bags
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # Team 0: +60 +2 bags. 7+2=9, no penalty
    check(new_bags[0] == 9, f"Team 0 bags = 9 (got {new_bags[0]})")
    check(deltas[0] == 62, f"Team 0 delta = 62 (got {deltas[0]})")

    # Now push to exactly 10
    game2 = new_game(target_score=500)
    game2.bags = [7, 0]

    ds2 = deal_cards(0, random.Random(1))
    ds2.bids = [3, 3, 3, 3]
    ds2.tricks_won = [5, 2, 4, 2]  # team 0 took 9, bid 6, +3 bags
    ds2.phase = 'done'

    deltas2, new_bags2 = score_deal(ds2, game2)
    # Team 0: +60 +3 bags. 7+3=10 → -100 penalty, bags reset to 0
    check(new_bags2[0] == 0, f"Team 0 bags reset to 0 (got {new_bags2[0]})")
    check(deltas2[0] == 63 - 100, f"Team 0 delta = -37 (got {deltas2[0]})")


def test_nil_bidder_tricks_count_for_team():
    print("test_nil_bidder_tricks_count_for_team")
    game = new_game(target_score=500)
    ds = deal_cards(0, random.Random(1))
    ds.bids = [0, 3, 4, 3]
    # P0 nil but took 1 trick, P2 bid 4 and took 5
    # Team 0 total tricks: 1 + 5 = 6
    ds.tricks_won = [1, 4, 5, 3]
    ds.phase = 'done'

    deltas, new_bags = score_deal(ds, game)
    # P0 nil failed: -100
    # Team 0 contract: P2 bid 4, team took 6 → +40 +2 bags = +42
    # Total: -100 + 42 = -58
    check(deltas[0] == -58, f"Nil failed + overtricks = -58 (got {deltas[0]})")
    check(new_bags[0] == 2, f"Team 0 bags = 2 (got {new_bags[0]})")


def test_ismcts_returns_legal_card():
    print("test_ismcts_returns_legal_card")
    from search import ismcts_choose

    rng = random.Random(42)
    game = new_game(target_score=500)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    p = ds.current_player
    legal = legal_plays(ds, p)

    # Run with few iterations for speed
    card = ismcts_choose(ds, game, p, iterations=50, rng=random.Random(99))
    check(card in legal, f"ISMCTS returned legal card {card!r}")
    check(card in ds.hands[p], f"ISMCTS card is in player's hand")


def test_ismcts_single_move():
    print("test_ismcts_single_move")
    from search import ismcts_choose

    rng = random.Random(42)
    game = new_game(target_score=500)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    # Play until someone has only 1 legal move
    found = False
    while ds.phase == 'playing' and not found:
        p = ds.current_player
        legal = legal_plays(ds, p)
        if len(legal) == 1:
            card = ismcts_choose(ds, game, p, iterations=10, rng=random.Random(1))
            check(card == legal[0], "Single legal move returned immediately")
            found = True
        play_card(ds, p, legal[0])

    if not found:
        print("  (skipped — no single-move situation found)")


def test_rulebot_plays_full_game():
    print("test_rulebot_plays_full_game")
    from play import RuleBot, play_game

    for seed in range(5):
        game = play_game(
            [RuleBot(), RuleBot(), RuleBot(), RuleBot()],
            target_score=200,
            verbose=False,
            seed=seed
        )
        check(game.game_over, f"Seed {seed}: game completed")
        check(game.winner in (0, 1), f"Seed {seed}: valid winner")
        # Verify score consistency
        winner = game.winner
        loser = 1 - winner
        if game.scores[loser] > -200:
            check(game.scores[winner] >= game.target_score,
                  f"Seed {seed}: winner reached target")


def test_nn_model_shapes():
    print("test_nn_model_shapes")
    try:
        import torch
        from nn import PlayModel, BidModel

        pm = PlayModel()
        x = torch.randn(2, FEATURE_DIM)
        policy, value = pm(x)
        check(policy.shape == (2, 52), f"Policy shape (2,52) got {policy.shape}")
        check(value.shape == (2, 1), f"Value shape (2,1) got {value.shape}")
        check(pm.param_count() > 0, f"Params: {pm.param_count():,}")

        bm = BidModel()
        x = torch.randn(2, BID_FEATURE_DIM)
        logits = bm(x)
        check(logits.shape == (2, 14), f"Bid logits shape (2,14) got {logits.shape}")

        print(f"  PlayModel: {pm.param_count():,} params")
        print(f"  BidModel: {sum(p.numel() for p in bm.parameters()):,} params")
    except ImportError:
        print("  (skipped — torch not installed)")


def test_feature_dim_consistency():
    print("test_feature_dim_consistency")
    # Verify FEATURE_DIM matches actual encoding
    rng = random.Random(42)
    game = new_game(target_score=500)
    ds = deal_cards(0, rng)
    for p in range(4):
        place_bid(ds, p, 3)

    # Play a few tricks to get interesting state
    for _ in range(8):  # 2 tricks
        if ds.phase != 'playing':
            break
        p = ds.current_player
        moves = legal_plays(ds, p)
        play_card(ds, p, moves[0])

    feat = encode_state(ds, game, 0)
    check(feat.shape[0] == FEATURE_DIM,
          f"FEATURE_DIM={FEATURE_DIM} matches encoding={feat.shape[0]}")

    bid_feat = encode_hand_for_bidding(ds.hands[0], 0, ds, game)
    check(bid_feat.shape[0] == BID_FEATURE_DIM,
          f"BID_FEATURE_DIM={BID_FEATURE_DIM} matches encoding={bid_feat.shape[0]}")


# --- Run all tests ---

if __name__ == '__main__':
    tests = [
        test_card_basics,
        test_deal,
        test_bidding,
        test_trick_resolution,
        test_legal_plays,
        test_follow_suit,
        test_spades_broken,
        test_full_deal,
        test_scoring_basic,
        test_scoring_failed,
        test_scoring_nil_success,
        test_scoring_nil_failed,
        test_scoring_bags_penalty,
        test_scoring_boston,
        test_scoring_boston_blocked_by_double_nil,
        test_scoring_double_nil_one_fails,
        test_game_win_condition,
        test_game_both_reach_target,
        test_game_negative_cutoff,
        test_full_game_completes,
        test_multiple_games_consistency,
        test_belief_void_tracking,
        test_belief_sampling,
        test_belief_sampling_with_voids,
        test_feature_encoding,
        test_bid_feature_encoding,
        test_bidding_heuristics,
        test_only_spades_can_lead,
        test_deal_rotation,
        test_cards_played_tracking,
        test_trick_winner_leads_next,
        test_no_duplicate_cards_in_play,
        test_follow_suit_enforced,
        test_scoring_edge_exact_bags,
        test_nil_bidder_tricks_count_for_team,
        test_ismcts_returns_legal_card,
        test_ismcts_single_move,
        test_rulebot_plays_full_game,
        test_nn_model_shapes,
        test_feature_dim_consistency,
    ]

    print(f"Running {len(tests)} tests...\n")

    for test in tests:
        try:
            test()
        except Exception as e:
            FAIL += 1
            print(f"  EXCEPTION in {test.__name__}: {e}")
        print()

    print(f"{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    if FAIL == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"{FAIL} TESTS FAILED ✗")
        sys.exit(1)
