import source.player as player
import re
import source.errors as errors
import numpy as np
from math import exp


class StackelbergAttacker(player.Attacker):
    """
    The Stackelberg attacker observes the Defender strategy and plays a pure
    strategy that best responds to it.
    """

    name = "stackelberg"
    pattern = re.compile(r"^" + name + "\d*$")

    def compute_strategy(self):
        return self.best_respond(self.game.strategy_history[-1])


class StackelbergAttackerR(player.Attacker):
    """
    The StackelbergR attacker observes the Defender strategy and plays a pure
    strategy that best responds to it. In order to break ties, it randomizes
    over the indifferent strategies.
    """

    name = "stackelbergR"
    pattern = re.compile(r"^" + name + "\d*$")

    def compute_strategy(self):
        return self.best_respond_mixed(self.game.strategy_history[-1])


class DumbAttacker(player.Attacker):
    """
    The Dumb attacker, given an initially choosen action, always plays itp
    """

    name = "dumb"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1, choice=None):
        super().__init__(game, id, resources)
        if not choice or len(choice) != self.resources:
            shuffled_targets = list(range(len(self.game.values)))
            np.random.shuffle(shuffled_targets)
            self.choice = shuffled_targets[:resources]

    def compute_strategy(self):
        targets = range(len(self.game.values))
        return [int(t in self.choice) for t in targets]


class FictitiousPlayerAttacker(player.Attacker):
    """
    The fictitious player computes the empirical distribution of the
    adversary move and then best respond to it. When it starts it has a vector
    of weights for each target and at each round the plays the inverse of that
    weight normalized to the weights sum. Then he observe the opponent's move
    and update the weights acconding to it.
    """
    name = "fictitious"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+))?$")

    def __init__(self, game, id, resources=1, initial_weight=10):
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


class StochasticAttacker(player.Attacker):
    """
    It attacks according to a fixed 
    """

    name = "stochastic_attacker"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            arguments = [float(a) for a in
                         player_type.split(cls.name)[1].split("-")
                         if a != '']
            if not arguments:
                return cls(game, id)
            elif len(arguments) == 1:
                return cls(game, id, int(arguments[0]))
            else:
                arguments[0] = int(arguments[0])
                if (len(arguments) == len(game.values) + 1):
                    is_prob = round(sum(arguments[1:]), 3) == 1
                    if is_prob:
                        args = [game, id] + arguments
                        return cls(*args)
                    else:
                        raise errors.NotAProbabilityError(arguments[1:])

    def __init__(self, game, id, resources=1, *distribution):
        super().__init__(game, id, resources)
        self.distribution = distribution
        targets = list(range(len(game.values)))
        if not distribution:
            self.distribution = [1 / len(targets) for t in targets]
        else:
            self.distribution = distribution

    def compute_strategy(self):
        return self.distribution


class QuantalResponseAttacker(player.Attacker):

    name = "QR"
    pattern = re.compile(r"^" + name + r"(\d+(-(\d+(\.\d+)?))?)?$")

    def __init__(self, game, id, resources=1, learning_rate=1):
        super().__init__(game, id, resources)
        self.learning_rate = learning_rate  # not used yet

    def compute_strategy(self):
        targets = range(len(self.game.values))
        strategies = self.game.strategy_history[-1]
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]
        not_norm_coverage = sum(defenders_strategies)

        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)
        values = np.array([self.game.values[t][self.id] for t in targets])
        expected_payoffs = values * (np.ones(len(targets)) - coverage)
        return self.quantal_response(expected_payoffs)

    def quantal_response(self, utilities):
        if not self.game.history:
            return self.br_uniform()
        weights = []
        time = len(self.game.history)
        # learning rate updated
        if time > 1:
            self.learning_rate = self.learning_rate / (time - 1) * time
        targets = list(range(len(self.game.values)))
        weights = np.array([exp(self.learning_rate * utilities[t])
                            for t in targets])
        weights = weights / np.linalg.norm(weights, ord=1)  # normalization
        return [float(w) for w in weights]
