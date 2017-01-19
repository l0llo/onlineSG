import source.player as player
import source.errors as errors
from math import exp, log, sqrt
from copy import copy, deepcopy
import re
import numpy as np
import enum
import gurobipy

Algorithm = enum.Enum('Algorithm', 'fpl wm fpls')


class FixedActionDefender(player.Defender):

    @classmethod
    def parse(cls, player_type, game, id):
        pass

    def __init__(self, game, id, action):
        super().__init__(game, id, 1)
        targets = list(range(len(self.game.values)))
        self.last_strategy = [int(t == action) for t in targets]

    def compute_strategy(self):
        return self.last_strategy


class KnownStochasticDefender(player.Defender):
    """
    """

    name = "stochastic_defender"
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
        targets = list(range(len(game.values)))
        self.br_stochastic_strategy = None
        if not distribution:
            self.distribution = [1 / len(targets) for t in targets]
        else:
            self.distribution = distribution

    def compute_strategy(self):
        return self.br_stochastic()

    def br_stochastic(self):  # what if indifferent?
        if not self.br_stochastic_strategy:
            targets = range(len(self.game.values))
            max_target = max(targets,
                             key=lambda x:
                             self.game.values[x][0] * self.distribution[x])
            self.stochastic_reward = (-self.game.values[max_target][0] *
                                      self.distribution[max_target])
            self.br_stochastic_strategy = [int(i == max_target)
                                           for i in targets]
        return self.br_stochastic_strategy


class StackelbergDefender(player.Defender):

    name = "stackelberg_defender"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1):
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


class ExpertDefender(player.Defender):

    @classmethod
    def parse(cls, player_type, game, id):
        return None

    def __init__(self, game, id, resources,
                 learning_rate, experts, algo='fpl'):
        super().__init__(game, id, resources)
        self.learning_rate = learning_rate
        self.experts = experts
        self.avg_rewards = {e: 0 for e in experts}
        self.norm_const = 1  # has to be initialized late
        self.algorithm = Algorithm[algo]
        #: id of the selected expert
        self.selected_expert = None
        self.t = 0

    def compute_strategy(self):
        """
        this do not allow  exposing mixed strategy at expert level:
        if you want to, modify this method in a subclass
        """
        if self.algorithm == Algorithm.fpl:
            exp_distribution = self.follow_the_perturbed_leader()
        elif self.algorithm == Algorithm.wm:
            exp_distribution = self.weighted_majority()
        elif self.algorithm == Algorithm.fpls:
            exp_distribution = self.fpl_with_sampling()
        self.selected_expert = self.experts[player.sample(exp_distribution, 1)[0]]
        for e in self.experts:
            if e != self.selected_expert:
                e.play_strategy()
        return self.selected_expert.play_strategy()

    def learn(self):
        self.learning_rate = self.learning_rate * max(self.t - 1, 1) / self.t
        for e in self.experts:
            moves = copy(self.game.history[-1])
            if e != self.selected_expert:
                a = e.sample_strategy()
                moves[0] = a
            current_reward = sum(self.game.get_player_payoffs(0, moves))
            self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.t - 1, 1) +
                                    current_reward) / self.t)

    def follow_the_perturbed_leader(self):
        if not self.game.history:
            self.norm_const = max([v[self.id] for v in self.game.values])
            return self.uniform_strategy(len(self.experts))
        perturbed_rewards = {e: 0 for e in self.experts}
        for e in self.experts:
            noise = np.random.uniform(0, self.norm_const * self.learning_rate)  # / log(time)
            perturbed_rewards[e] = self.avg_rewards[e] + noise  # +: we have a reward, not of a loss
        perturbed_leader = max(self.experts,
                               key=lambda e: perturbed_rewards[e])
        return [int(e == perturbed_leader) for e in self.experts]

    def weighted_majority(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.experts))
        weights = []
        for e in self.experts:
            weights.append(np.array(exp(self.learning_rate *
                                        self.avg_rewards[e])))
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
            return self.uniform_strategy(len(self.experts))
        samples_sum = np.zeros(len(self.experts))
        for i in range(iterations):
            sample = self.follow_the_perturbed_leader()
            samples_sum += np.array(sample)
        samples_sum /= iterations
        return [float(s) for s in samples_sum]

    def receive_feedback(self, feedback):
        if feedback:
            self.feedbacks.append(feedback)
        if self.t > 0:
            self.learn()
        for e in self.experts:
            e.receive_feedback(None)
        self.t += 1

    def _json(self):
        self_copy = deepcopy(self)
        d = self_copy.__dict__
        d.pop("game", None)
        d.pop("avg_rewards", None)
        d["experts"] = [str(e) for e in self.experts]
        d["algorithm"] = d["algorithm"].name
        d["class_name"] = self.__class__.__name__
        return d


class MABDefender(ExpertDefender):
    """
    Learns in a Multi Armed Bandit way: only the selected expert (arm) can 
    observe the feedback of the chosen action
    """

    @classmethod
    def parse(cls, player_type, game, id):
        return None

    def __init__(self, game, id, resources,
                 learning_rate, experts):
        super().__init__(game, id, resources, learning_rate, experts)
        self.weight = {e: 0 for e in experts}
        N = len(self.experts)
        T = self.game.time_horizon
        self.prob = {e: 1 / N for e in self.experts}
        self.beta = sqrt((N * log(N)) / ((exp(1) - 1) * T))

    def learn(self):
        # self.learning_rate = self.learning_rate * max(self.t - 1, 1) / self.t
        # moves = copy(self.game.history[-1])
        # cur_reward = sum(self.game.get_player_payoffs(0, moves))
        # # translate and normalize to be in [0, 1]
        # cur_reward = (cur_reward + self.norm_const) / self.norm_const
        e = self.selected_expert
        # N = len(self.experts)
        # eta = sqrt((N * log(N)) / ((exp(1) - 1) * self.t))
        # self.weight[e] = self.weight[e] * exp(eta * cur_reward /
        #                                       self.prob[e])
        moves = copy(self.game.history[-1])
        current_reward = sum(self.game.get_player_payoffs(0, moves))
        self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.weight[e], 1) +
                                current_reward) / (self.weight[e] + 1))
        self.weight[e] += 1

    def compute_strategy(self):
        if not self.game.history:
            self.norm_const = max([v[self.id] for v in self.game.values])
        #exp_distribution = self.exp3()
        exp_distribution = self.ucb1()
        self.selected_expert = self.experts[player.sample(exp_distribution, 1)[0]]
        return self.selected_expert.play_strategy()

    def exp3(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.experts))
        norm = np.linalg.norm(np.array(list(self.weight.values())), ord=1)
        for e in self.experts:
            self.prob[e] = ((1 - self.beta) * self.weight[e] / norm +
                            self.beta / len(self.experts))
        return [float(self.prob[e]) for e in self.experts]

    def ucb1(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.experts))
        r = dict()
        for e in self.experts:
            b = sqrt(log(self.t) / max(self.weight[e], 1))
            r[e] = self.avg_rewards[e] + b
        max_e = max(self.experts, key=lambda e: r[e])
        return [int(e == max_e) for e in self.experts]

    def receive_feedback(self, feedback):
        if feedback:
            self.feedbacks.append(feedback)
        #if self.t > 0:
        self.learn()
        self.selected_expert.receive_feedback(None)
        self.t += 1

    def _json(self):
        d = super()._json()
        d.pop("weight", None)
        d.pop("prob", None)
        return d


class UnknownStochasticDefender(ExpertDefender):
    """
    """

    name = "unknown_stochastic_defender"
    pattern = re.compile(r"^" + name + r"(\d+(-(\d+(\.\d+)?))?)?$")

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

    def __init__(self, game, id, resources=1, learning_rate=1,
                 algorithm='fpls'):
        # experts = [FixedActionDefender(game, self.id, i)
        #            for i in list(range(len(game.values)))]
        experts = list(range(len(game.values)))
        super().__init__(game, id, resources, learning_rate, experts, algorithm)

    def compute_strategy(self):
        if self.algorithm == Algorithm.fpl:
            return self.follow_the_perturbed_leader()
        elif self.algorithm == Algorithm.wm:
            return self.weighted_majority()
        elif self.algorithm == Algorithm.fpls:
            return self.fpl_with_sampling()

    def learn(self):
        self.learning_rate = self.learning_rate * max(self.t - 1, 1) / self.t
        for e in self.experts:
            moves = copy(self.game.history[-1])
            moves[0] = [e]
            current_reward = sum(self.game.get_player_payoffs(0, moves))
            self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.t - 1, 1) +
                                    current_reward) / self.t)

    def receive_feedback(self, feedback):
        if feedback:
            self.feedbacks.append(feedback)
        if self.t > 0:
            self.learn()
        self.t += 1
