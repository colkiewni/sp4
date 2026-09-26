"""
Evaluate new model vs current best.
Run on VPS after each training cycle.
Usage: python evaluate.py --new models/play_model_candidate.onnx --games 200
On PROMOTE, the candidate is copied to models/play_model.onnx (the canonical
"current best" file). On REJECT, models/play_model.onnx is left untouched —
train.py never writes it directly, only evaluate.py does, on promotion.
"""

from __future__ import annotations
import argparse
import random
import os
import time
import multiprocessing as mp
from engine import new_game, start_deal, finish_deal, place_bid, play_card, legal_plays
from bidding import make_bid
from search import ismcts_choose_with_policy
from nn_inference import PlayModelInference, load_models, get_play_model, make_nn_policy_fn, make_nn_value_fn
from play import RuleBot, SearchBot, play_game
import numpy as np


class NNSearchBot:
    """SearchBot guided by a specific NN model."""
    def __init__(self, model: PlayModelInference, iterations: int = 1000):
        self.model = model
        self.iterations = iterations
        self._game = None  # set before each deal

    def set_game(self, game):
        self._game = game

    def choose_bid(self, hand, player, deal, game):
        return make_bid(hand, player, deal, game)

    def choose_card(self, player, deal, game):
        policy_fn = make_nn_policy_fn(self.model, game)
        # ponytail: value_fn (1-ply NN cutoff instead of full rollout) is disabled
        # for now — diagnosed empirically: with it, this bot won 15% vs RuleBot;
        # policy-only (full rollouts, like plain search) won 68%. The value head
        # isn't trained enough yet to beat honest rollouts. Re-enable
        # (value_fn=make_nn_value_fn(self.model, game, deal.team_of(player)))
        # once more training data brings its val loss down enough to re-test.
        card, _ = ismcts_choose_with_policy(
            deal, game, player,
            policy_fn=policy_fn,
            value_fn=None,
            iterations=self.iterations
        )
        return card


# --- Per-worker state: each pool process loads the model ONCE and reuses it
# across all games it's handed, instead of reloading per game. ---
_worker_new_bot = None
_worker_baseline_bot = None


def _init_worker(new_model_path: str, baseline: str, search_iters: int,
                  baseline_model_path: str = None):
    """Pool initializer: runs once per worker process."""
    global _worker_new_bot, _worker_baseline_bot
    model = PlayModelInference(new_model_path)
    _worker_new_bot = NNSearchBot(model, iterations=search_iters)
    if baseline_model_path:
        # Model-vs-model: pit the new candidate against a SPECIFIC other
        # checkpoint (e.g. last cycle's promoted model), instead of a fixed
        # rule/search baseline. Answers "did this get better or worse than
        # last time" directly, rather than inferring it from two separate
        # noisy comparisons against an unrelated baseline.
        baseline_model = PlayModelInference(baseline_model_path)
        _worker_baseline_bot = NNSearchBot(baseline_model, iterations=search_iters)
    else:
        _worker_baseline_bot = RuleBot() if baseline == 'rule' else SearchBot(iterations=search_iters)


def _play_eval_game(args: tuple) -> dict:
    """Play one evaluation game in a worker process."""
    game_index, target_score, seed_offset = args
    if game_index % 2 == 0:
        players = [_worker_new_bot, _worker_baseline_bot, _worker_new_bot, _worker_baseline_bot]
        new_team = 0
    else:
        players = [_worker_baseline_bot, _worker_new_bot, _worker_baseline_bot, _worker_new_bot]
        new_team = 1

    game = play_game(players, target_score=target_score, verbose=False,
                      seed=seed_offset + game_index)

    return {
        'new_won': game.winner == new_team,
        'score_diff': game.scores[new_team] - game.scores[1 - new_team],
    }


def evaluate(
    new_model_path: str,
    baseline: str = 'rule',
    num_games: int = 200,
    target_score: int = 500,
    search_iters: int = 1000,
    seed_offset: int = 0,
    num_workers: int = None,
    baseline_model_path: str = None
) -> dict:
    """
    Pit new NN model (team 0) against baseline (team 1).
    Games are distributed across worker processes (multiprocessing.Pool),
    same pattern as selfplay.py, each worker loading the model once.
    Returns win rates and score stats.

    baseline_model_path: when given, ignores `baseline` and instead pits the
    new model against another specific .onnx checkpoint (both using NNSearchBot).
    Use this to directly compare successive self-play cycles against each
    other, rather than each separately against rule/search.
    """
    if not os.path.exists(new_model_path):
        raise FileNotFoundError(f"Model not found: {new_model_path}")
    if new_model_path.endswith('.pt'):
        raise ValueError(
            f"evaluate.py loads models via onnxruntime and can't read a raw .pt "
            f"checkpoint ({new_model_path}). Point --new at the matching .onnx "
            f"export instead, e.g. models/play_model.onnx (train.py writes this "
            f"automatically alongside the .pt file)."
        )
    if baseline_model_path:
        if not os.path.exists(baseline_model_path):
            raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}")
        if baseline_model_path.endswith('.pt'):
            raise ValueError(f"--baseline-model needs an .onnx file, not {baseline_model_path}")
    elif baseline not in ('rule', 'search'):
        raise ValueError(f"Unknown baseline: {baseline}")

    num_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
    num_workers = min(num_workers, num_games)  # no idle workers on small runs
    print_every = max(1, num_games // 20)  # ~20 updates total, works for small runs too

    # Team 0 = new model (players 0, 2)
    # Team 1 = baseline (players 1, 3)
    wins = [0, 0]
    score_diffs = []
    args_list = [(i, target_score, seed_offset) for i in range(num_games)]

    t0 = time.time()
    with mp.Pool(num_workers, initializer=_init_worker,
                 initargs=(new_model_path, baseline, search_iters, baseline_model_path)) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_eval_game, args_list)):
            if result['new_won']:
                wins[0] += 1
            else:
                wins[1] += 1
            score_diffs.append(result['score_diff'])

            if (i + 1) % print_every == 0 or (i + 1) == num_games:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60  # games/min
                eta_min = (num_games - i - 1) / (i + 1) * elapsed / 60
                wr = wins[0] / (i + 1)
                print(f"  Game {i+1}/{num_games} | Win rate: {wr:.3f} | "
                      f"Avg score diff: {np.mean(score_diffs):.1f} | "
                      f"{rate:.1f} games/min | ETA: {eta_min:.1f}min")

    total = wins[0] + wins[1]
    win_rate = wins[0] / total

    result = {
        'new_model_wins': wins[0],
        'baseline_wins': wins[1],
        'win_rate': win_rate,
        'avg_score_diff': float(np.mean(score_diffs)),
        'std_score_diff': float(np.std(score_diffs)),
        'promoted': win_rate > 0.53,  # promote if >53% win rate
    }

    print(f"\n{'='*50}")
    baseline_label = f"model {baseline_model_path}" if baseline_model_path else baseline
    print(f"Evaluation: {num_games} games vs {baseline_label}")
    print(f"  New model wins: {wins[0]} ({win_rate:.1%})")
    print(f"  Baseline wins:  {wins[1]}")
    print(f"  Avg score diff: {result['avg_score_diff']:.1f} ± {result['std_score_diff']:.1f}")
    print(f"  Decision: {'PROMOTE ✓' if result['promoted'] else 'REJECT ✗'}")

    if result['promoted']:
        import shutil
        best_path = os.path.join(os.path.dirname(new_model_path), 'play_model.onnx')
        # Convert .pt to .onnx if needed
        if new_model_path.endswith('.pt'):
            import torch
            from nn import PlayModel, export_onnx
            from features import FEATURE_DIM
            model_pt = PlayModel()
            model_pt.load_state_dict(torch.load(new_model_path, map_location='cpu'))
            export_onnx(model_pt, best_path, FEATURE_DIM)
            print(f"  Exported promoted model → {best_path}")
        elif os.path.abspath(new_model_path) == os.path.abspath(best_path):
            # --new already points at the canonical file (e.g. re-evaluating an
            # already-promoted model) — nothing to promote it TO, it's already there.
            print(f"  Already the canonical model at {best_path}, nothing to copy")
        else:
            shutil.copy(new_model_path, best_path)
            print(f"  Copied promoted model → {best_path}")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate new model vs baseline')
    parser.add_argument('--new', type=str, required=True,
                        help='Path to candidate .onnx file (e.g. models/play_model_candidate.onnx)')
    parser.add_argument('--baseline', type=str, default='rule',
                        choices=['rule', 'search'],
                        help='Baseline to compare against (ignored if --baseline-model is given)')
    parser.add_argument('--baseline-model', type=str, default=None,
                        help='Compare against another specific .onnx checkpoint instead of '
                             'rule/search — e.g. an older promoted model, to directly measure '
                             'whether a new cycle actually improved on the last one.')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--target', type=int, default=500)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel game workers (default: cpu_count-1)')
    args = parser.parse_args()

    evaluate(
        new_model_path=args.new,
        baseline=args.baseline,
        num_games=args.games,
        target_score=args.target,
        search_iters=args.iters,
        seed_offset=args.seed,
        num_workers=args.workers,
        baseline_model_path=args.baseline_model
    )
