# from random import uniform, shuffle
import random
import gurobipy
import numpy as np
import re


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
    pattern = re.compile(r"^" + name + "\d?$")

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


class Defender(Player):
    name = "defender"
    pattern = re.compile(r"^" + name + "\d?$")

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


class Attacker(Player):
    """
    The Attacker base class from which all the attacker inherit: it implements
    the best_respond method which is used by many types of adversaries.
    """

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
    """
    The Stackelberg attacker observes the Defender strategy and plays a pure
    strategy that best responds to it.
    """

    name = "stackelberg"
    pattern = re.compile(r"^" + name + "\d?$")

    def compute_strategy(self):
        return self.best_respond(self.game.strategy_history[-1])


class DumbAttacker(Attacker):
    """
    The Dumb attacker, given an initially choosen action, always plays itp
    """

    name = "dumb"
    pattern = re.compile(r"^" + name + "\d?$")

    def __init__(self, game, id, resources=1, choice=None):
        super().__init__(game, id, resources)
        if not choice or len(choice) != self.resources:
            shuffled_targets = list(range(len(self.game.values)))
            random.shuffle(shuffled_targets)
            self.choice = shuffled_targets[:resources]

    def compute_strategy(self):
        targets = range(len(self.game.values))
        return [int(t in self.choice) for t in targets]


class FictiousPlayerAttacker(Attacker):
    """
    The fictitious player computes the empirical distribution of the adversary
    move and then best respond to it. When it starts it has a vector of weights
    for each target and at each round the plays the inverse of that weight 
    normalized to the weights sum. Then he observe the opponent's move and 
    update the weights acconding to it.
    """
    name = "fictitious"
    pattern = re.compile(r"^" + name + r"\d?(-\d)?$")

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


class StUDefender(Defender):
    """
    This defender is able to distinguish between a uniform
    or a stackelberg attacker and best respond accordingly

    This is only an example: from our computation in fact against these two
    kind of adversaries there is no interests in distinguish between them:
    in fact playing always the best response to a stackelberg player does not
    generate any regret.
    """
    name = "stu_defender"
    pattern = re.compile(r"^" + name + "\d?$")

    def __init__(self, game, id, resources=1, confidence=0.9):
        super().__init__(game, id, resources)
        self.confidence = confidence
        self.mock_stackelberg = StackelbergAttacker(self.game, 1)
        self.belief = {'uniform': 1,
                       'stackelberg': 0}

    def compute_strategy(self):
        if len(self.game.history) == 1 or self.belief['stackelberg']:
            last_move = self.game.history[-1][1][0]
            mock_move = [i for (i, s) in enumerate(self.mock_stackelberg.compute_strategy())
                         if s][0]
            if last_move == mock_move:
                self.belief['uniform'] = self.belief['uniform'] * (1 / len(self.game.values))
                self.belief['stackelberg'] = 1 - self.belief['uniform']
            else:
                self.belief['uniform'] = 1
                self.belief['stackelberg'] = 0

        if self.belief['stackelberg']:
            return self.br_stackelberg()  # minimax in two players game
        else:
            return self.br_uniform()  # highest value action

    def br_uniform(self):
        targets = range(len(self.game.values))
        max_target = max(targets, key=lambda x: self.game.values[x][0])
        return [int(i == max_target) for i in targets]

    def br_stackelberg(self):
        m = gurobipy.Model("SSG")
        targets = list(range(len(self.game.values)))
        strategy = []
        for t in targets:
            strategy.append(m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name="x" + str(t)))
        v = m.addVar(lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name="v")
        m.setObjective(v, gurobipy.GRB.MAXIMIZE)
        for t in targets:
            terms = [-self.game.values[t][self.id] * strategy[i]
                     for i in targets if i != t]
            m.addConstr(sum(terms) - v >= 0, "c" + str(t))
        m.addConstr(sum(strategy) == 1, "c" + str(len(targets)))
        m.optimize()
        return [float(s.x) for s in strategy]
