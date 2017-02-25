import source.player as player
import source.errors as errors
import source.standard_player_parsers as spp
from math import exp, log, sqrt
from copy import copy, deepcopy
import re
import numpy as np
import enum
import gurobipy

ExpAlgorithm = enum.Enum('ExpAlgorithm', 'fpl wm fpls')


class FixedActionDefender(player.Defender):

    @classmethod
    def parse(cls, player_type, game, id):
        pass

    def __init__(self, game, id, action):
        super().__init__(game, id, 1)
        targets = list(range(len(self.game.values)))
        self.fixed_strategy = [int(t == action) for t in targets]

    def compute_strategy(self, strategy=None):
        if strategy:
            return strategy
        return self.fixed_strategy


class KnownStochasticDefender(player.Defender):
    """
    """

    name = "sto_def"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.stochastic_parse(cls, player_type, game, id)

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
            self.stochastic_reward = sum([(-self.game.values[i][0] *
                                           self.distribution[i])
                                          for i in targets
                                          if i != max_target])
            self.br_stochastic_strategy = [int(i == max_target)
                                           for i in targets]
        return self.br_stochastic_strategy


class StackelbergDefender(player.Defender):

    name = "sta_def"
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

    name = "expert"

    @classmethod
    def parse(cls, player_type, game, id):
        return None

    def __init__(self, game, id, resources,
                 learning_rate, algo='fpl', *arms):
        super().__init__(game, id, resources)
        if isinstance(algo, int):
            self.algorithm = ExpAlgorithm(algo)
        else:
            self.algorithm = ExpAlgorithm[algo]
        self.learning_rate = learning_rate
        self.arms = arms
        self.avg_rewards = None
        self.norm_const = 1  # has to be initialized late
        self.learning = player.Learning.EXPERT
        #: id of the selected expert
        self.sel_arm = None

    def compute_strategy(self):
        """
        this do not allow  exposing mixed strategy at expert level:
        if you want to, modify this method in a subclass
        """
        if self.tau == 0:
            self.avg_rewards = {e: 0 for e in self.arms}
        if self.algorithm == ExpAlgorithm.fpl:
            exp_distribution = self.follow_the_perturbed_leader()
        elif self.algorithm == ExpAlgorithm.wm:
            exp_distribution = self.weighted_majority()
        elif self.algorithm == ExpAlgorithm.fpls:
            exp_distribution = self.fpl_with_sampling()
        self.sel_arm = self.arms[player.sample(exp_distribution, 1)[0]]
        for e in self.arms:
            if e != self.sel_arm:
                e.play_strategy()
        return self.sel_arm.play_strategy()

    def learn(self):
        if self.tau > 0:
            self.learning_rate = self.learning_rate * max(self.tau - 1, 1) / self.tau
            for e in self.arms:
                moves = copy(self.game.history[-1])
                if e != self.sel_arm:
                    a = e.sample_strategy()
                    moves[0] = a
                # if this expert defender has not played the real move
                elif (e.last_strategy !=
                      self.game.strategy_history[-1][self.id]):
                    a = e.sample_strategy()
                    moves[0] = a
                current_reward = sum(self.game.get_player_payoffs(0, moves))
                self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.tau - 1, 1) +
                                        current_reward) / self.tau)

    def follow_the_perturbed_leader(self):
        if not self.game.history:
            self.norm_const = max([v[self.id] for v in self.game.values])
            return self.uniform_strategy(len(self.arms))
        perturbed_rewards = {e: 0 for e in self.arms}
        for e in self.arms:
            noise = np.random.uniform(0, self.norm_const * self.learning_rate)  # / log(time)
            perturbed_rewards[e] = self.avg_rewards[e] + noise  # +: we have a reward, not of a loss
        perturbed_leader = max(self.arms,
                               key=lambda e: perturbed_rewards[e])
        return [int(e == perturbed_leader) for e in self.arms]

    def weighted_majority(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.arms))
        weights = []
        for e in self.arms:
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
            return self.uniform_strategy(len(self.arms))
        samples_sum = np.zeros(len(self.arms))
        for i in range(iterations):
            sample = self.follow_the_perturbed_leader()
            samples_sum += np.array(sample)
        samples_sum /= iterations
        return [float(s) for s in samples_sum]

    def _json(self):
        self_copy = deepcopy(self)
        d = self_copy.__dict__
        d.pop("game", None)
        d.pop("avg_rewards", None)
        d["arms"] = [str(e) for e in self.arms]
        d["algorithm"] = d["algorithm"].name
        d["class_name"] = self.__class__.__name__
        d.pop("learning", None)
        return d


class MABDefender(ExpertDefender):
    """
    Learns in a Multi Armed Bandit way: only the selected expert (arm) can 
    observe the feedback of the chosen action
    """

    name = "mab"

    @classmethod
    def parse(cls, player_type, game, id):
        return None

    def __init__(self, game, id, resources, *arms):
        super().__init__(game, id, resources, 1, 'fpl', *arms)
        self.learning = player.Learning.MAB
        self.weight = {e: 0 for e in arms}
        self.prob = None
        self.beta = None

    def learn(self):
        # moves = copy(self.game.history[-1])
        # cur_reward = sum(self.game.get_player_payoffs(0, moves))
        # # translate and normalize to be in [0, 1]
        # cur_reward = (cur_reward + self.norm_const) / self.norm_const
        e = self.sel_arm
        # N = len(self.arms)
        # eta = sqrt((N * log(N)) / ((exp(1) - 1) * self.tau))
        # self.weight[e] = self.weight[e] * exp(eta * cur_reward /
        #                                       self.prob[e])
        moves = copy(self.game.history[-1])
        current_reward = sum(self.game.get_player_payoffs(0, moves))
        self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.weight[e], 1) +
                                current_reward) / (self.weight[e] + 1))
        self.weight[e] += 1

    def compute_strategy(self):
        if self.tau == 0:
            self.weight = {e: 0 for e in self.arms}
            N = len(self.arms)
            T = self.game.time_horizon
            self.prob = {e: 1 / N for e in self.arms}
            self.beta = sqrt((N * log(N)) / ((exp(1) - 1) * T))
            self.norm_const = max([v[self.id] for v in self.game.values])
            self.avg_rewards = {e: 0 for e in self.arms}
        #exp_distribution = self.exp3()
        exp_distribution = self.ucb1()
        self.sel_arm = self.arms[player.sample(exp_distribution, 1)[0]]
        return self.sel_arm.play_strategy()

    def exp3(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.arms))
        norm = np.linalg.norm(np.array(list(self.weight.values())), ord=1)
        for e in self.arms:
            self.prob[e] = ((1 - self.beta) * self.weight[e] / norm +
                            self.beta / len(self.arms))
        return [float(self.prob[e]) for e in self.arms]

    def ucb1(self):
        if not self.game.history:
            return self.uniform_strategy(len(self.arms))
        r = dict()
        for e in self.arms:
            b = sqrt(log(self.tau) / max(self.weight[e], 1))
            r[e] = (self.avg_rewards[e] / self.norm_const) + b
        max_e = max(self.arms, key=lambda e: r[e])
        return [int(e == max_e) for e in self.arms]

    def _json(self):
        d = super()._json()
        d.pop("weight", None)
        d.pop("prob", None)
        return d


class UnknownStochasticDefender(ExpertDefender):
    """
    we should go back to the version with FixedActionDefender
    to uniform learning notation
    """

    name = "usto_def"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.stochastic_parse(cls, player_type, game, id)

    def __init__(self, game, id, resources=1, learning_rate=1,
                 algorithm='fpls'):
        arms = list(range(len(game.values)))
        super().__init__(game, id, resources, learning_rate,
                         algorithm, *arms)

    def compute_strategy(self):
        if self.algorithm == ExpAlgorithm.fpl:
            return self.follow_the_perturbed_leader()
        elif self.algorithm == ExpAlgorithm.wm:
            return self.weighted_majority()
        elif self.algorithm == ExpAlgorithm.fpls:
            return self.fpl_with_sampling()

    def learn(self):
        self.learning_rate = self.learning_rate * max(self.tau - 1, 1) / self.tau
        for e in self.arms:
            moves = copy(self.game.history[-1])
            moves[0] = [e]
            current_reward = sum(self.game.get_player_payoffs(0, moves))
            self.avg_rewards[e] = ((self.avg_rewards[e] * max(self.tau - 1, 1) +
                                    current_reward) / self.tau)

    def receive_feedback(self, feedback):
        """
        has to be revised: receive_feedback should not be modified
        we
        """
        if feedback:
            self.feedbacks.append(feedback)
        if self.tau > 0:
            self.learn()
        self.tau += 1


class UnknownStochasticDefender2(ExpertDefender):
    """
    we should go back to the version with FixedActionDefender
    to uniform learning notation
    """

    name = "usto_defV2"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.stochastic_parse(cls, player_type, game, id)

    def __init__(self, game, id, resources=1, learning_rate=1,
                 algorithm='fpls'):
        arms = [FixedActionDefender(game, id, t)
                for t in range(len(game.values))]
        super().__init__(game, id, resources, learning_rate,
                         arms, algorithm)
        import source.players.attackers as attackers
        self.mock_sto = attackers.MockStochasticAttacker(game, 1, resources)

    def learn(self):
        self.learning_rate = (self.learning_rate * max(self.tau - 1, 1) /
                              max(self.tau, 1))
        self.avg_rewards = dict()
        self.mock_sto.play_strategy()
        targets = list(range(len(self.game.values)))
        for i in targets:
            strategies = {0: [int(i == t) for t in targets],
                          1: self.mock_sto.last_strategy}
            self.avg_rewards[self.arms[i]] = -self.mock_sto.exp_loss(strategies)