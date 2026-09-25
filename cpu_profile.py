"""
Find the best worker configuration for your specific VPS.
Run once: python cpu_profile.py
"""

import time
import os
import multiprocessing as mp
import random

def _time_n_games(args):
    """Time N games with given config."""
    num_games, search_iters, ismcts_workers = args

    # Import here so it works in subprocess
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from selfplay import play_full_game_selfplay

    t0 = time.time()
    for i in range(num_games):
        play_full_game_selfplay(
            target_score=200,
            search_iterations=search_iters,
            seed=i,
            collect_data=False,
            ismcts_workers=ismcts_workers
        )
    return time.time() - t0


def profile():
    cpu_count = os.cpu_count() or 4
    print(f"CPU cores available: {cpu_count}")
    print(f"Profiling different configurations...\n")

    # Test configs: (game_workers, ismcts_workers_per_game)
    configs = []
    for gw in [1, 2, 4, cpu_count // 2, cpu_count - 1, cpu_count]:
        for iw in [1, 2]:
            if gw * iw <= cpu_count + 2:  # allow slight oversubscription
                configs.append((gw, iw))

    # Remove duplicates and sort
    configs = sorted(set(configs))

    GAMES_PER_TEST = 4
    ITERS = 300
    TARGET = 200

    results = []

    for game_workers, ismcts_workers in configs:
        total_cores = game_workers * ismcts_workers
        print(f"Testing: {game_workers} game workers × {ismcts_workers} ISMCTS workers "
              f"= {total_cores} cores...", end=' ', flush=True)

        t0 = time.time()

        if game_workers == 1:
            # Single process — no pool overhead
            elapsed = _time_n_games((GAMES_PER_TEST, ITERS, ismcts_workers))
        else:
            # Split games across workers
            games_each = max(1, GAMES_PER_TEST // game_workers)
            actual_games = games_each * game_workers

            args_list = [(games_each, ITERS, ismcts_workers)] * game_workers
            with mp.Pool(game_workers) as pool:
                times = pool.map(_time_n_games, args_list)
            elapsed = max(times)  # wall time = slowest worker

        games_per_min = GAMES_PER_TEST / elapsed * 60
        print(f"{games_per_min:.1f} games/min")
        results.append((game_workers, ismcts_workers, games_per_min))

    print(f"\n{'='*60}")
    print(f"{'Config':<35} {'Games/min':>10}")
    print(f"{'-'*60}")

    results.sort(key=lambda x: x[2], reverse=True)
    for gw, iw, gpm in results:
        marker = " ← BEST" if results.index((gw, iw, gpm)) == 0 else ""
        print(f"  {gw} game workers × {iw} ISMCTS workers    {gpm:>8.1f}{marker}")

    best_gw, best_iw, best_gpm = results[0]
    print(f"\nRecommended command:")
    print(f"  python selfplay.py --games 1000 --iterations 1000 "
          f"--workers {best_gw} --ismcts-workers {best_iw} --output selfplay_data")

    print(f"\nEstimated throughput at 1000 iterations:")
    # Scale from 300 iters to 1000 iters (roughly linear)
    scaled = best_gpm * (300 / 1000)
    print(f"  ~{scaled:.1f} games/min")
    print(f"  ~{scaled * 60 * 24:.0f} games/day")
    print(f"  ~{scaled * 60 * 24 * 400:.0f} play samples/day")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # safe on all platforms
    profile()
