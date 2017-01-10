import source.player as player
import source.players.attackers as attackers
import source.players.base_defenders as base_defenders
import source.errors as errors
from math import log, sqrt, exp
import re
import gurobipy
import numpy as np


class StUDefender(player.Defender):
    """
    This defender is able to distinguish between a uniform
    or a stackelberg attacker and best respond accordingly
    """
    name = "stu_defender"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1, confidence=0.9):
        super().__init__(game, id, resources)
        self.confidence = confidence
        self.mock_stackelberg = attackers.StackelbergAttacker(self.game, 1)
        self.belief = {'uniform': 1,
                       'stackelberg': 0}
        self.br_stackelberg_strategy = None

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

    def br_stackelberg(self):
        if not self.br_stackelberg_strategy:
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
            m.params.outputflag = 0
            m.optimize()
            self.br_stackelberg_strategy = [float(s.x) for s in strategy]
        return self.br_stackelberg_strategy


class AWESOMS_UCB(base_defenders.StackelbergDefender,
                  base_defenders.KnownStochasticDefender):
    """
    Adapt When Everybody is Stochastic, Otherwise Move to Stackelberg
    with Upper Confidence Bound where p=t^(-4)

    Defender detection of Stackelberg-Known Stochastic
    """

    name = "awesoms_ucb"
    pattern = re.compile(r"^" + name + r"((\d+(\.\d+)?)+(-(\d+(\.\d+)?)+)+)?$")

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
        player.Defender.__init__(self, game, id, resources)
        self.distribution = distribution
        self.br_stackelberg_strategy = None
        self.maxmin = None
        self.stackelberg = True

    def compute_strategy(self):
        # if it is the first round then br to stackelberg
        t = len(self.game.history)
        if t < 2:
            return self.br_stackelberg()
        else:
            # compute the new bound
            if self.stackelberg:
                bound = sqrt(2 * log(t) / t)
                # check if it has been exceeded
                average_reward = sum([sum(f.values()) for f in self.feedbacks]) / t
                self.stackelberg = (self.maxmin - average_reward <= bound)
                if self.stackelberg:
                    return self.br_stackelberg()
            return self.br_stochastic()


class USD(base_defenders.UnknownStochasticDefender):

    name = "usd"
    pattern = re.compile(r"^" + name + r"-(fpl|fpls|wm)$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            algorithm = player_type.split(cls.name)[1].split("-")[1]
            args = [game, id, algorithm]
            return cls(*args)

    def __init__(self, game, id, algorithm="fpl"):
        super().__init__(game, id)
        self.algorithm = algorithm

    def compute_strategy(self):
        if self.game.history:
            self.update_experts()
        if self.algorithm == "fpl":
            return self.follow_the_perturbed_leader()
        elif self.algorithm == "fpls":
            return self.fpl_with_sampling()
        elif self.algorithm == "wm":
            return self.weighted_majority()


class USD2(base_defenders.UnknownStochasticDefender):

    name = "usd2"
    pattern = re.compile(r"^" + name + r"-(fpl|fpls|wm)$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            algorithm = player_type.split(cls.name)[1].split("-")[1]
            args = [game, id, algorithm]
            return cls(*args)

    def __init__(self, game, id, algorithm="fpl"):
        super().__init__(game, id)
        self.algorithm = algorithm

    def compute_strategy(self):
        if self.game.history:
            self.update_experts()
        t = len(self.game.history)
        if self.algorithm == "fpl":
            return self.follow_the_perturbed_leader()
        elif self.algorithm == "fpls":
            if t > 2:
                self.learning_rate = self.learning_rate * sqrt(t - 1) / sqrt(t)
            return self.fpl_with_sampling()
        elif self.algorithm == "wm":
            return self.weighted_majority()

    def weighted_majority(self):  # not great results..
        if not self.game.history:
            return self.br_uniform()
        weights = []
        time = len(self.game.history)
        # learning rate updated
        if time > 1:
            self.learning_rate = self.learning_rate / (sqrt(time - 1)) * sqrt(time)
        experts = list(range(len(self.game.values)))
        for e in experts:
            avg_reward = self.expert_tot_rewards[e] / time
            weights.append(np.array(exp(self.learning_rate * avg_reward)))
        weights = np.array(weights)
        weights = weights / np.linalg.norm(weights, ord=1)  # normalization
        return [float(w) for w in weights]