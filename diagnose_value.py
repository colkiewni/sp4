"""
One-off diagnostic: NN policy prior only, value_fn=None so every simulation
still does a full rollout (like plain search), isolating whether the POLICY
or the VALUE head is responsible for the win-rate drop vs plain ISMCTS.
Run: python diagnose_value.py --model models/play_model.onnx --games 50
"""
import argparse
from nn_inference import PlayModelInference
from search import ismcts_choose_with_policy
from play import RuleBot, play_game, Player
from evaluate import make_nn_policy_fn


class NNPolicyOnlyBot(Player):
    def __init__(self, model, iterations=1000):
        self.model = model
        self.iterations = iterations

    def choose_bid(self, hand, player, deal, game):
        from bidding import make_bid
        return make_bid(hand, player, deal, game)

    def choose_card(self, player, deal, game):
        policy_fn = make_nn_policy_fn(self.model, game)
        card, _ = ismcts_choose_with_policy(
            deal, game, player,
            policy_fn=policy_fn, value_fn=None,  # <- key difference: no value cutoff
            iterations=self.iterations
        )
        return card


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/play_model.onnx')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--iters', type=int, default=1000)
    args = parser.parse_args()

    model = PlayModelInference(args.model)
    bot = NNPolicyOnlyBot(model, iterations=args.iters)
    rule = RuleBot()

    wins = [0, 0]
    for i in range(args.games):
        if i % 2 == 0:
            players = [bot, rule, bot, rule]; team = 0
        else:
            players = [rule, bot, rule, bot]; team = 1
        game = play_game(players, target_score=200, verbose=False, seed=i)
        wins[0 if game.winner == team else 1] += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.games} | policy-only-NN win rate: {wins[0]/(i+1):.3f}")

    print(f"\nFinal: policy-only-NN {wins[0]}/{args.games} ({100*wins[0]/args.games:.1f}%) vs RuleBot")
