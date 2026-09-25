"""
Self-play game generation for training data.
Runs on VPS CPU. Produces (state, policy, outcome) tuples.
"""

from __future__ import annotations
import random
import numpy as np
from typing import Optional
from engine import (
    Card, DealState, GameState, new_game, start_deal, finish_deal,
    place_bid, play_card, legal_plays, legal_bids, score_deal
)
from bidding import make_bid
from search import ismcts_choose_with_policy, ismcts_choose
from features import encode_state, encode_hand_for_bidding, FEATURE_DIM, BID_FEATURE_DIM, CARD_INDEX
from copy import deepcopy
import os
import time
import multiprocessing as mp



def run_selfplay(
    num_games: int = 100,
    target_score: int = 500,
    search_iterations: int = 1000,
    num_workers: int = None,
    output_dir: str = 'selfplay_data',
    ismcts_workers_per_game: int = 1
):
    """
    Run parallel self-play games.

    Two levels of parallelism:
      Level 1: num_workers games run simultaneously (multiprocessing.Pool)
      Level 2: within each game, ismcts_workers_per_game cores for ISMCTS

    Total cores used = num_workers × ismcts_workers_per_game
    Recommendation for 16 vCPU:
      - num_workers=8, ismcts_workers_per_game=2  (16 cores, balanced)
      - num_workers=14, ismcts_workers_per_game=1 (14 cores, max throughput)
      - num_workers=4, ismcts_workers_per_game=4  (16 cores, stronger search)
    """
    cpu_count = os.cpu_count() or 4

    if num_workers is None:
        # Leave 1 core free for OS. Split remaining between game-level parallelism.
        # Default: maximize game throughput (1 ISMCTS worker per game)
        num_workers = max(1, cpu_count - 1)

    total_cores = num_workers * ismcts_workers_per_game
    if total_cores > cpu_count:
        print(f"Warning: {total_cores} total cores requested but only {cpu_count} available.")
        print(f"  Reducing num_workers to {max(1, cpu_count // ismcts_workers_per_game)}")
        num_workers = max(1, cpu_count // ismcts_workers_per_game)

    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (i, target_score, search_iterations, output_dir, ismcts_workers_per_game)
        for i in range(num_games)
    ]

    print(f"{'='*60}")
    print(f"Spades Self-Play Data Generation")
    print(f"  Games:            {num_games}")
    print(f"  ISMCTS iters:     {search_iterations}/move")
    print(f"  Game workers:     {num_workers}")
    print(f"  ISMCTS workers:   {ismcts_workers_per_game}/game")
    print(f"  Total cores used: {num_workers * ismcts_workers_per_game}/{cpu_count}")
    print(f"  Output:           {output_dir}/")
    print(f"{'='*60}")

    t0 = time.time()
    results = []
    total_play_samples = 0
    wins = [0, 0]

    with mp.Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, args_list)):
            results.append(result)
            if result.get('winner') is not None:
                wins[result['winner']] += 1
            if 'play_features' in result:
                total_play_samples += len(result['play_features'])

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60  # games per minute
            eta = (num_games - i - 1) / ((i + 1) / elapsed)

            print(f"  [{i+1:4d}/{num_games}] "
                  f"W:{result.get('winner','?')} "
                  f"Score:{result.get('scores','?')} "
                  f"Deals:{result.get('deals','?'):2d} | "
                  f"{rate:.1f} games/min | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Samples: {total_play_samples:,}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Time:          {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Throughput:    {num_games/elapsed*60:.1f} games/min")
    print(f"  Play samples:  {total_play_samples:,}")
    print(f"  Team 0 wins:   {wins[0]} ({100*wins[0]/num_games:.1f}%)")
    print(f"  Team 1 wins:   {wins[1]} ({100*wins[1]/num_games:.1f}%)")
    print(f"  Output size:   {_dir_size_mb(output_dir):.1f} MB")
    print(f"{'='*60}")

    return results


def _dir_size_mb(path: str) -> float:
    total = 0
    for f in os.scandir(path):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def _worker(args):
    """Worker function for parallel self-play."""
    game_id, target_score, search_iterations, output_dir, ismcts_workers = args
    seed = (int(time.time() * 1000) % (2**31) + game_id * 7919) % (2**31)

    samples, game = play_full_game_selfplay(
        target_score=target_score,
        search_iterations=search_iterations,
        seed=seed,
        collect_data=True,
        ismcts_workers=ismcts_workers
    )

    play_features, play_policies, play_values = [], [], []
    bid_features, bid_targets, bid_values = [], [], []

    for s in samples:
        if s['type'] == 'play':
            play_features.append(s['features'])
            play_policies.append(s['policy'])
            play_values.append(s['value'])
        elif s['type'] == 'bid':
            bid_features.append(s['features'])
            bid_targets.append(s['bid'])
            bid_values.append(s['value'])

    result = {
        'game_id': game_id,
        'winner': game.winner,
        'scores': game.scores,
        'deals': game.deal_number,
    }

    save_data = {}
    if play_features:
        save_data['play_features'] = np.array(play_features, dtype=np.float32)
        save_data['play_policies'] = np.array(play_policies, dtype=np.float32)
        save_data['play_values'] = np.array(play_values, dtype=np.float32)
        result['play_features'] = save_data['play_features']

    if bid_features:
        save_data['bid_features'] = np.array(bid_features, dtype=np.float32)
        save_data['bid_targets'] = np.array(bid_targets, dtype=np.int64)
        save_data['bid_values'] = np.array(bid_values, dtype=np.float32)

    if save_data and output_dir:
        path = os.path.join(output_dir, f'game_{game_id:06d}.npz')
        np.savez_compressed(path, **save_data)

    return result


def play_full_game_selfplay(
    target_score: int = 500,
    search_iterations: int = 1000,
    seed: int = None,
    collect_data: bool = True,
    ismcts_workers: int = 1
) -> tuple[list[dict], GameState]:
    """Play a full game, return all training samples and final game state."""
    rng = random.Random(seed)
    game = new_game(target_score=target_score, first_dealer=rng.randint(0, 3))
    all_samples = []

    while not game.game_over:
        samples, game = play_one_deal_selfplay(
            game,
            search_iterations=search_iterations,
            rng=rng,
            collect_data=collect_data,
            ismcts_workers=ismcts_workers
        )
        all_samples.extend(samples)

    return all_samples, game


def play_one_deal_selfplay(
    game: GameState,
    search_iterations: int = 1000,
    rng: random.Random = None,
    collect_data: bool = True,
    ismcts_workers: int = 1
) -> tuple[list[dict], GameState]:
    """Play one deal with ISMCTS, collecting training data."""
    rng = rng or random.Random()
    game = start_deal(game, rng)
    deal = game.deal
    samples = []

    # Bidding phase
    while deal.phase == 'bidding':
        p = deal.current_player
        hand = set(deal.hands[p])
        if collect_data:
            bid_features = encode_hand_for_bidding(hand, p, deal, game)
        bid = make_bid(hand, p, deal, game)
        if collect_data:
            samples.append({
                'type': 'bid',
                'features': bid_features,
                'bid': bid,
                'player': p,
                'team': deal.team_of(p),
            })
        place_bid(deal, p, bid)

    # Playing phase
    while deal.phase == 'playing':
        p = deal.current_player
        team = deal.team_of(p)
        if collect_data:
            state_features = encode_state(deal, game, p)

        moves = legal_plays(deal, p)
        if len(moves) == 1:
            card = moves[0]
            visit_dist = {card: 1.0}
        else:
            card, visit_dist = ismcts_choose_with_policy(
                deal, game, p,
                iterations=search_iterations,
                rng=rng,
                num_workers=ismcts_workers
            )

        if collect_data:
            policy_target = np.zeros(52, dtype=np.float32)
            for c, prob in visit_dist.items():
                policy_target[CARD_INDEX[c]] = prob
            samples.append({
                'type': 'play',
                'features': state_features,
                'policy': policy_target,
                'player': p,
                'team': team,
            })

        play_card(deal, p, card)

    # Score and assign value targets
    deltas, _ = score_deal(deal, game)
    for sample in samples:
        t = sample['team']
        diff = deltas[t] - deltas[1 - t]
        sample['value'] = max(-1.0, min(1.0, diff / 100.0))

    game = finish_deal(game)

    if game.game_over and game.winner is not None:
        for sample in samples:
            t = sample['team']
            sample['value'] = 1.0 if t == game.winner else -1.0

    return samples, game


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spades self-play data generation')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--target', type=int, default=500)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=None,
                        help='Game-level workers (default: cpu_count-1)')
    parser.add_argument('--ismcts-workers', type=int, default=1,
                        help='ISMCTS workers per game (default: 1)')
    parser.add_argument('--output', type=str, default='selfplay_data')
    args = parser.parse_args()

    run_selfplay(
        num_games=args.games,
        target_score=args.target,
        search_iterations=args.iterations,
        num_workers=args.workers,
        output_dir=args.output,
        ismcts_workers_per_game=args.ismcts_workers
    )
