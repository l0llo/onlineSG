from random import uniform, shuffle
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


class Defender(Player):

    def __init__(self, game, id, resources):
        """"
        Attributes

        feedbacks   list of targets dict with feedbacks for each turn
                    (if any)
        """
        super().__init__(game, id, resources)
        self.feedbacks = []

    def receive_feedback(self, feedback):
        self.feedbacks.append(feedback)


class Attacker(Player):

    def best_respond(self, strategies):
        """
        Compute the pure strategy that best respond to a given dict of
        defender strategies
        """
        targets = range(len(self.game.values))

        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]

        # (sum the probabilities of differents defenders)
        not_norm_coverage = sum(defenders_strategies)

        # normalize
        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)

        # compute the expected value of each target (v[t]*(1-c[t]))
        values = np.array([self.game.values[t][self.id] for t in targets])
        expected_payoffs = values * (np.ones(len(targets)) - coverage)

        # play the argmax
        selected_targets = sorted(targets,
                                  key=lambda t: expected_payoffs[t],
                                  reverse=True)[:self.resources]
        return [int(t in selected_targets) for t in targets]


class StackelbergAttacker(Attacker):

    def compute_strategy(self):
        return self.best_respond(self.game.strategy_history[-1])


class DumbAttacker(Attacker):

    def __init__(self, game, id, resources, choice=None):
        super().__init__(game, id, resources)
        if not choice or len(choice) != self.resources:
            shuffled_targets = list(range(len(self.game.values)))
            shuffle(shuffled_targets)
            self.choice = shuffled_targets[:resources]

    def compute_strategy(self):
        targets = range(len(self.game.values))
        return [int(t in self.choice) for t in targets]


class FictiousPlayerAttacker(Attacker):

    def __init__(self, game, id, resources, initial_weight=10):
        super().__init__(game, id, resources)
        self.weights = None
        self.initial_weight = initial_weight

    def compute_strategy(self):
        """
        Add 1 to the weight of each covered target in the defender profile
        at each round: then best respond to the computed strategy
        """
        if self.game.history:
            for d in self.game.defenders:
                for t in self.game.history[-1][d]:
                    self.weights[d][t] += 1
        else:
            targets = range(len(self.game.values))
            self.weights = {d: [self.initial_weight for t in targets]
                            for d in self.game.defenders}
        return self.best_respond(self.weights)
