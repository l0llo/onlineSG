from source.player import Attacker
import re
import random


class StackelbergAttacker(Attacker):
    """
    The Stackelberg attacker observes the Defender strategy and plays a pure
    strategy that best responds to it.
    """

    name = "stackelberg"
    pattern = re.compile(r"^" + name + "\d*$")

    def compute_strategy(self):
        return self.best_respond(self.game.strategy_history[-1])


class DumbAttacker(Attacker):
    """
    The Dumb attacker, given an initially choosen action, always plays itp
    """

    name = "dumb"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1, choice=None):
        super().__init__(game, id, resources)
        if not choice or len(choice) != self.resources:
            shuffled_targets = list(range(len(self.game.values)))
            random.shuffle(shuffled_targets)
            self.choice = shuffled_targets[:resources]

    def compute_strategy(self):
        targets = range(len(self.game.values))
        return [int(t in self.choice) for t in targets]


class FictitiousPlayerAttacker(Attacker):
    """
    The fictitious player computes the empirical distribution of the adversary
    move and then best respond to it. When it starts it has a vector of weights
    for each target and at each round the plays the inverse of that weight 
    normalized to the weights sum. Then he observe the opponent's move and 
    update the weights acconding to it.
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
