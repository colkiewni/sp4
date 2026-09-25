"""
Information Set Monte Carlo Tree Search (ISMCTS) for card play decisions.

Parallelism strategy:
  - Game-level: selfplay.py runs N games simultaneously via multiprocessing.Pool
  - Within a single game decision: single-process UCB tree (correct and fast)
  - Parallel ISMCTS within one decision is complex and was causing the bug
    ponytail: if single-decision latency ever matters more than throughput,
    revisit root parallelization with proper virtual loss
"""

from __future__ import annotations
import math
import random
from copy import deepcopy
from typing import Optional

from engine import (
    Card, Suit, DealState, GameState,
    legal_plays, play_card, score_deal
)
from belief import BeliefState, make_belief


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ISMCTSNode:
    __slots__ = ('card', 'parent', 'children', 'visits', 'value', 'player')

    def __init__(
        self,
        card: Optional[Card],
        player: int,
        parent: Optional['ISMCTSNode'] = None
    ):
        self.card = card
        self.parent = parent
        self.children: list[ISMCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.player = player

    def ucb1(self, explore: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + explore * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child_visits(self) -> 'ISMCTSNode':
        return max(self.children, key=lambda c: c.visits)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _rollout_deal(deal: DealState, rng: random.Random) -> DealState:
    """Play out the rest of the deal with random legal moves."""
    while deal.phase == 'playing':
        p = deal.current_player
        moves = legal_plays(deal, p)
        if not moves:
            break
        play_card(deal, p, rng.choice(moves))
    return deal


def _evaluate_terminal(deal: DealState, game: GameState, team: int) -> float:
    """
    Evaluate a completed deal from team's perspective.
    Returns value in [-1, 1].
    """
    deltas, _ = score_deal(deal, game)
    my_new  = game.scores[team]     + deltas[team]
    opp_new = game.scores[1 - team] + deltas[1 - team]
    target  = game.target_score

    if my_new >= target and my_new > opp_new:   return  1.0
    if opp_new >= target and opp_new > my_new:  return -1.0
    if my_new  <= -200:                          return -1.0
    if opp_new <= -200:                          return  1.0

    diff = deltas[team] - deltas[1 - team]
    return max(-1.0, min(1.0, diff / 100.0))


def _determinize(
    deal: DealState,
    observer: int,
    belief: BeliefState,
    rng: random.Random
) -> DealState:
    """Sample a world consistent with belief state."""
    det = deepcopy(deal)
    sampled = belief.sample_consistent_deal(deal, rng)
    for p in range(4):
        if p != observer:
            det.hands[p] = sampled[p]
    return det


# ---------------------------------------------------------------------------
# Core ISMCTS — single process, proper UCB tree
# ---------------------------------------------------------------------------

def ismcts_choose(
    deal: DealState,
    game: GameState,
    player: int,
    iterations: int = 3000,
    rng: random.Random = None,
    num_workers: int = None   # kept for API compat, ignored
) -> Card:
    """
    Run ISMCTS and return the best card to play.

    Single-process UCB tree — correct and fast.
    Game-level parallelism (multiple games at once) is handled by selfplay.py.
    """
    rng = rng or random.Random()
    team = deal.team_of(player)
    belief = make_belief(player, deal)

    moves = legal_plays(deal, player)
    if len(moves) == 1:
        return moves[0]

    root = ISMCTSNode(card=None, player=player)

    for _ in range(iterations):
        # 1. Determinize: sample a consistent world
        det = _determinize(deal, player, belief, rng)

        # 2. Find legal moves in this determinization
        det_moves = legal_plays(det, player)
        if not det_moves:
            continue

        # 3. Sync tree with this determinization
        #    Add any new legal cards not yet in tree
        existing = {c.card for c in root.children}
        for card in det_moves:
            if card not in existing:
                root.children.append(
                    ISMCTSNode(card=card, player=player, parent=root)
                )

        # 4. Select among children compatible with this determinization
        compatible = [c for c in root.children if c.card in det_moves]
        if not compatible:
            continue

        unvisited = [c for c in compatible if c.visits == 0]
        if unvisited:
            chosen = rng.choice(unvisited)
        else:
            chosen = max(compatible, key=lambda c: c.ucb1())

        # 5. Simulate: play chosen card then random rollout
        sim = deepcopy(det)
        play_card(sim, player, chosen.card)
        _rollout_deal(sim, rng)

        # 6. Evaluate
        value = _evaluate_terminal(sim, game, team)

        # 7. Backpropagate
        chosen.visits += 1
        chosen.value  += value
        root.visits   += 1

    # Return most-visited legal card
    legal_set = set(moves)
    candidates = [c for c in root.children if c.card in legal_set and c.visits > 0]
    if not candidates:
        return rng.choice(moves)

    return max(candidates, key=lambda c: c.visits).card


# ---------------------------------------------------------------------------
# ISMCTS with NN guidance — for selfplay training data collection
# ---------------------------------------------------------------------------

def ismcts_choose_with_policy(
    deal: DealState,
    game: GameState,
    player: int,
    policy_fn=None,
    value_fn=None,
    iterations: int = 3000,
    rng: random.Random = None,
    num_workers: int = None   # ignored, kept for API compat
) -> tuple[Card, dict[Card, float]]:
    """
    ISMCTS with optional NN policy prior and value function.
    Returns (best_card, visit_distribution) for training targets.
    """
    rng = rng or random.Random()
    team = deal.team_of(player)
    belief = make_belief(player, deal)

    moves = legal_plays(deal, player)
    if len(moves) == 1:
        return moves[0], {moves[0]: 1.0}

    # Get policy prior if NN available
    prior: dict[Card, float] = {}
    if policy_fn:
        try:
            prior = policy_fn(deal, player)
        except Exception:
            pass

    root = ISMCTSNode(card=None, player=player)

    for _ in range(iterations):
        det = _determinize(deal, player, belief, rng)
        det_moves = legal_plays(det, player)
        if not det_moves:
            continue

        # Sync tree
        existing = {c.card for c in root.children}
        for card in det_moves:
            if card not in existing:
                root.children.append(
                    ISMCTSNode(card=card, player=player, parent=root)
                )

        compatible = [c for c in root.children if c.card in det_moves]
        if not compatible:
            continue

        unvisited = [c for c in compatible if c.visits == 0]

        if unvisited:
            if prior:
                weights = [max(prior.get(c.card, 0.01), 1e-6) for c in unvisited]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                chosen = rng.choices(unvisited, weights=weights, k=1)[0]
            else:
                chosen = rng.choice(unvisited)
        else:
            if prior:
                total_v = root.visits or 1
                def puct(node):
                    q = node.value / node.visits
                    p = max(prior.get(node.card, 0.01), 1e-6)
                    u = 2.0 * p * math.sqrt(total_v) / (1 + node.visits)
                    return q + u
                chosen = max(compatible, key=puct)
            else:
                chosen = max(compatible, key=lambda c: c.ucb1())

        sim = deepcopy(det)
        play_card(sim, player, chosen.card)

        if value_fn and sim.phase == 'playing':
            value = value_fn(sim, team)
        else:
            _rollout_deal(sim, rng)
            value = _evaluate_terminal(sim, game, team)

        chosen.visits += 1
        chosen.value  += value
        root.visits   += 1

    # Build visit distribution over legal moves
    legal_set = set(moves)
    candidates = [c for c in root.children if c.card in legal_set and c.visits > 0]

    if not candidates:
        return rng.choice(moves), {m: 1.0 / len(moves) for m in moves}

    total_visits = sum(c.visits for c in candidates)
    visit_dist = {c.card: c.visits / total_visits for c in candidates}

    best = max(candidates, key=lambda c: c.visits)
    return best.card, visit_dist
