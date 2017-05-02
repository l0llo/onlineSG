import source.player as player
import source.standard_player_parsers as spp
from math import exp, log, sqrt
from copy import copy, deepcopy
from source.errors import AlreadyFinalizedError
import numpy as np
import re


class Expert(player.Defender):

    name = "EXP"
    pattern = re.compile(r"^" + name + r"\d$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources):
        super().__init__(game, id, resources)
        self.avg_rewards = None
        self.norm_const = 1
        self.sel_p = None
        self.last_p_strategies = None

    def finalize_init(self):
        super().finalize_init()
        self.norm_const = max([v[self.id] for v in self.game.values])
        self.avg_rewards = {p: 0 for p in self.A}
        self.last_p_strategies = {p: None for p in self.A}

    def compute_strategy(self):
        self.sel_p = self.follow_the_perturbed_leader()
        for p in self.A:
            self.last_p_strategies[p] = self.br_to(p)
        return self.last_p_strategies[self.sel_p]

    def learn(self):
        for p in self.A:
            str_dict = {0: self.last_p_strategies[p],
                        1: self.ps(self.game.history[-1][1][0])}
            cur_reward = -p.exp_loss(str_dict)
            self.avg_rewards[p] = ((self.avg_rewards[p] * (self.tau() - 1) +
                                    cur_reward) / self.tau())

    def follow_the_perturbed_leader(self):
        """
        returns the chosen profile
        """

        def noise():
            return np.random.uniform(0, (self.norm_const * sqrt(len(self.A)) /
                                         (self.tau() + 1)))
        perturbed_leader = max(self.A,
                               key=lambda p: self.avg_rewards[p] + noise())
        return perturbed_leader

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


class MAB(player.Defender):
    """
    Learns in a Multi Armed Bandit way: only the selected expert (arm) can 
    observe the feedback of the chosen action
    """

    name = "MAB"
    pattern = re.compile(r"^" + name + r"\d$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources):
        super().__init__(game, id, 1)
        self.weight = None
        self.sel_p = None
        # self.prob = None
        # self.beta = None

    def finalize_init(self):
            super().finalize_init()
            self.weight = {p: 0 for p in self.A}
            # N = len(self.arms)
            # T = self.game.time_horizon
            # self.prob = {e: 1 / N for e in self.arms}
            # self.beta = sqrt((N * log(N)) / ((exp(1) - 1) * T))
            self.norm_const = max([v[self.id] for v in self.game.values])
            self.avg_rewards = {p: 0 for p in self.A}

    def compute_strategy(self):
        self.sel_p = self.ucb1()
        return self.br_to(self.sel_p)

    def learn(self):
        p = self.sel_p
        cur_reward = sum(self.game.get_last_turn_payoffs(0))
        self.avg_rewards[p] = ((self.avg_rewards[p] * max(self.weight[p], 1) +
                                cur_reward) / (self.weight[p] + 1))
        self.weight[p] += 1

    # def exp3(self):
    #     if not self.game.history:
    #         return self.uniform_strategy(len(self.arms))
    #     norm = np.linalg.norm(np.array(list(self.weight.values())), ord=1)
    #     for e in self.arms:
    #         self.prob[e] = ((1 - self.beta) * self.weight[e] / norm +
    #                         self.beta / len(self.arms))
    #     return [float(self.prob[e]) for e in self.arms]

    def ucb1(self):
        r = dict()
        if not self.tau():
            choice = np.random.choice(list(range(len(self.A))))
            return self.A[choice]
        else:
            for p in self.A:
                b = sqrt(log(self.tau()) / max(self.weight[p], 1))
                r[p] = (self.avg_rewards[p] / self.norm_const) + b
            max_p = max(self.A, key=lambda p: r[p])
            return max_p

    def _json(self):
        d = super()._json()
        d.pop("weight", None)
        d.pop("prob", None)
        return d
