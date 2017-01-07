import random
import numpy as np
import re
from copy import deepcopy


class Player:
    """
    It is the base class from which all players inherit. It implements the 
    default compute_strategy method (as a uniform player) and the 
    sample_strategy method.

    Each subclass has class attributes (name and pattern) and a class method
    (parse): they are used for the parsing of the player columns in the config 
    files.
    """
    name = "player"
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        """
        This is the default
        """
        if cls.pattern.match(player_type):
            args = [game, id] + [int(a) for a in
                                 player_type.split(cls.name)[1].split("-")
                                 if a != '']
            return cls(*args)
        else:
            return None

    def __init__(self, game, id, resources=1):
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
        sample = [random.uniform(0, strategy[i]) for i in targets]
        selected_targets = sorted(targets, key=lambda k: sample[k],
                                  reverse=True)[:self.resources]
        return selected_targets

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " id:", str(self.id),
                        " resources:", str(self.resources), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " id:", str(self.id),
                        " resources:", str(self.resources), ">"])

    def _json(self):
        self_copy = deepcopy(self)
        d = self_copy.__dict__
        d.pop('game', None)
        d["class_name"] = self.__class__.__name__
        return d


class Defender(Player):
    name = "defender"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1):
        """"
        Attributes

        feedbacks   list of targets dict with feedbacks for each turn
                    (if any)
        """
        super().__init__(game, id, resources)
        self.feedbacks = []

    def receive_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def last_reward(self):
        return sum(self.feedbacks[-1].values())

    def br_uniform(self):
        targets = range(len(self.game.values))
        max_target = max(targets, key=lambda x: self.game.values[x][0])
        return [int(i == max_target) for i in targets]


class Attacker(Player):
    """
    The Attacker base class from which all the attacker inherit: it implements
    the best_respond method which is used by many types of adversaries.
    """

    def best_respond(self, strategies):
        """
        Compute the pure strategy that best respond to a given dict of
        defender strategies. 
        Should it randomize over indifferent maximum actions?
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
