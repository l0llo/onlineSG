from random import uniform
import numpy as np


class Player:

    def __init__(self, game, id, resources):
        """

        """
        self.game = game
        self.id = id
        self.resources = resources

    def compute_strategy(self):
        """
        set a probability distribution over the targets
        default: uniform strategy
        """
        targets_number = len(self.game.values)
        return [self.resources / targets_number for i in range(targets_number)]

    def sample_strategy(self):
        """
        sample a move from the computed distribution
        """
        targets = range(len(self.game.values))
        strategy = self.game.strategy_history[-1][self.id]
        sample = [uniform(0, strategy[i]) for i in targets]
        selected_targets = sorted(targets, key=lambda k: sample[k],
                                  reverse=True)[:self.resources]
        return selected_targets

    def get_rewards(self, t=None):
        """
        return all the rewards until t-turn
        """
        if not t:
            t = len(self.game.history)
        return list(map(lambda m: self.game.get_player_payoff(self.id, m),
                        self.game.history[:t]))


class StackelbergAttacker(Player):

    def compute_strategy(self):
        targets = range(len(self.game.values))

        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(self.game.strategy_history[-1][d])
                                for d in self.game.defenders]

        # (sum the probabilities of differents defenders)
        not_normalized_coverage = sum(defenders_strategies)

        # normalize
        coverage = not_normalized_coverage / np.linalg.norm(not_normalized_coverage, ord=1)

        # compute the expected value of each target (v[t]*(1-c[t]))
        values = np.array([self.game.values[t][self.id] for t in targets])
        expected_payoffs = values * (np.ones(len(targets)) - coverage)

        # play the argmax
        choice = max(targets, key=lambda t: expected_payoffs[t])
        return [int(t==choice) for t in targets]


