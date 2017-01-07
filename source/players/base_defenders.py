import source.player as player
from source.errors import NotAProbabilityError
from math import exp
from copy import copy
import random
import re
import numpy as np
import gurobipy


class KnownStochasticDefender(player.Defender):
    """
    """

    name = "stochastic_defender"
    pattern = re.compile(r"^" + name + r"((\d+(\.\d+)?)+(-(\d+(\.\d+)?))+)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            arguments = [float(a) for a in
                         player_type.split(cls.name)[1].split("-")
                         if a != '']
            arguments[0] = int(arguments[0])
            if (len(arguments) == len(game.values) + 1):
                is_prob = round(sum(arguments[1:]), 3) == 1
                if is_prob:
                    args = [game, id] + arguments
                    return cls(*args)
                else:
                    raise NotAProbabilityError(arguments[1:])

    def __init__(self, game, id, resources, *distribution):
        super().__init__(game, id, resources)
        self.distribution = distribution

    def compute_strategy(self):
        return self.br_stochastic()

    def br_stochastic(self):  # what if indifferent?
        targets = range(len(self.game.values))
        max_target = max(targets,
                         key=lambda x:
                         self.game.values[x][0] * self.distribution[x])
        return [int(i == max_target) for i in targets]


class UnknownStochasticDefender(player.Defender):
    """
    implementing Follow the Perturbed Leader
    -> PROBLEM: being a pure strategy, Stackelberg Attacker can exploits it!
    """

    name = "unknown_stochastic_defender"
    pattern = re.compile(r"^" + name + r"(\d+(-(\d+(\.\d+)?))?)?$")

    def __init__(self, game, id, resources=1, learning_rate=0.5):
        super().__init__(game, id, resources)
        self.learning_rate = learning_rate  # not used yet
        experts = list(range(len(self.game.values)))
        self.expert_tot_rewards = [0 for e in experts]
        self.norm_const = 1  # has to be initialized later

    def compute_strategy(self):
        # return self.weighted_majority()
        if self.game.history:
            self.update_experts()
        # return self.fpl_with_sampling()
        return self.follow_the_perturbed_leader()

    def update_experts(self):
        experts = list(range(len(self.game.values)))
        moves = copy(self.game.history[-1])
        for e in experts:
            moves[0] = [e]
            current_reward = sum(self.game.get_player_payoffs(0, moves))
            self.expert_tot_rewards[e] += current_reward

    def follow_the_perturbed_leader(self):
        if not self.game.history:
            self.norm_const = max([v[self.id] for v in self.game.values])
            return self.br_uniform()
        experts = list(range(len(self.game.values)))
        perturbed_rewards = [0 for e in experts]
        #  time = len(self.game.history)
        for e in experts:
            noise = random.uniform(0, self.norm_const)  # / log(time)
            perturbed_rewards[e] = self.expert_tot_rewards[e] + noise  # +: we have a reward, not of a loss
        perturbed_leader = max(experts, key=lambda e: perturbed_rewards[e])
        return [int(e == perturbed_leader) for e in experts]

    def weighted_majority(self):
        if not self.game.history:
            return self.br_uniform()
        weights = []
        time = len(self.game.history)
        # learning rate updated
        if time > 1:
            self.learning_rate = self.learning_rate / (time - 1) * time
        experts = list(range(len(self.game.values)))
        for e in experts:
            avg_reward = self.expert_tot_rewards[e] / time
            weights.append(np.array(exp(self.learning_rate * avg_reward)))
        weights = np.array(weights)
        weights = weights / np.linalg.norm(weights, ord=1)  # normalization
        return [float(w) for w in weights]

    def fpl_with_sampling(self, iterations=1000):
        """
        mixed strategy version obtained with sampling: it is possible to
        optimize it adding some checks
        """
        if not self.game.history:
            self.norm_const = max([v[self.id] for v in self.game.values])
            return self.br_uniform()
        samples_sum = np.zeros(len(self.game.values))
        for i in range(iterations):
            sample = self.follow_the_perturbed_leader()
            samples_sum += np.array(sample)
        samples_sum /= iterations
        return [float(s) for s in samples_sum]


class StackelbergDefender(player.Defender):
    def __init__(self, game, id, resources=1, confidence=0.9):
        super().__init__(game, id, resources)
        self.br_stackelberg_strategy = None
        self.maxmin = None

    def compute_strategy(self):
        return self.br_stackelberg()

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
            self.maxmin = v.x
            self.br_stackelberg_strategy = [float(s.x) for s in strategy]
        return self.br_stackelberg_strategy
