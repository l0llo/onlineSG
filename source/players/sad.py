import source.player as player
import source.players.base_defenders as base_defenders
import source.players.attackers as attackers
import source.standard_player_parsers as spp
from math import log, sqrt
from copy import copy
from functools import reduce
import enum
import re


Detection = enum.Enum('Detection', 'strategy_aware not_strategy_aware')


class StrategyAwareDetector(base_defenders.StackelbergDefender):
    """
    only for non-history-based advesaries!
    """

    name = "sad"
    pattern = re.compile(r"^" + name + r"\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    def __init__(self, g, id, resources):
        super().__init__(g, id, resources)
        self.K2 = None
        self.K = None
        self.belief = None
        self.strategy_aware = None
        self.detection = None
        self.exp_defender = None
        self.learning = player.Learning.EXPERT
        self.arms = None

    def finalize_init(self):
        self.K = self.game.get_profiles_copies()
        self.belief = {k: 1 / (len(self.K)) for k in self.K}
        self.strategy_aware = [k for k in self.K
                               if (k.__class__ ==
                                   attackers.StackelbergAttacker)][0]
        self.K2 = copy(self.K)
        self.K2.remove(self.strategy_aware)
        experts = [k.get_best_responder() for k in self.K2]
        self.exp_defender = (base_defenders.
                             ExpertDefender(self.game, self.id,
                                            self.resources, 1,
                                            'fpl', *experts))
        self.arms = [self.exp_defender]

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = {k: k.last_strategy[o] * self.belief[k] for k in self.K}
        eta = 1 / sum(update.values())
        update = {k: update[k] * eta for k in update}  # normalization
        return update

    def find_t(self):
        """
        returns the t which maximizes the future belief of s.a.
        in case the prevision is true.
        """
        targets = list(range(len(self.game.values)))
        future_belief = dict()
        strategy = self.br_stackelberg()
        sel_targets = [t for t in targets if strategy[t] > 0]
        for t in sel_targets:
            self.strategy_aware.last_strategy = [int(t == i)
                                                 for i in targets]
            for k in self.K2:
                k.play_strategy()
            future_belief[t] = self.update_belief(t)
        sa_beliefs = [future_belief[t][self.strategy_aware]
                      for t in sel_targets]
        m = max(sa_beliefs)
        possible_t = [t for t in sel_targets
                      if future_belief[t][self.strategy_aware] == m]
        # target = np.random.choice(possible_t)
        target = possible_t[0]
        return target

    def compute_strategy(self):
        if self.detection is None:
            self.exp_defender.play_strategy()
            t = self.find_t()
            targets = list(range(len(self.game.values)))
            strategy = copy(self.br_stackelberg())
            sel_targets = [i for i in targets if strategy[i] > 0]
            epsilon = min(strategy[t] for t in sel_targets) / 100
            for i, s in enumerate(strategy):
                if i == t:
                    strategy[i] -= epsilon
                elif i in sel_targets:
                    strategy[i] += epsilon / (len(sel_targets) - 1)
            return strategy
        elif self.detection == Detection.strategy_aware:
            return self.br_stackelberg()
        elif self.detection == Detection.not_strategy_aware:
            return self.exp_defender.compute_strategy()

    def learn(self):
        if self.detection is None:
            for k in self.K:
                k.play_strategy()
            conditions = []
            if self.tau() > 2:
                exceeded = []
                targets = list(range(len(self.game.values)))
                for i in self.K2:
                    # print(i.distribution)
                    conditions = []
                    for t in targets:
                        # how many times he has played adv_moves
                        n = len([h for h in self.game.history
                                 if h[1][0] == t])
                        sample_mean = n / (self.tau())
                        # print(sample_mean)
                        p = i.last_strategy[t]  # STOCHASTIC ASSUMPTION!!!!!
                        conditions.append((abs(p - sample_mean) / 2) >
                                          sqrt(2 * log(self.tau()) /
                                               (self.tau())))
                        # print(conditions[-1])
                    exceeded.append(reduce(lambda a, b: a or b, conditions))
                if reduce(lambda a, b: a and b, exceeded):
                    self.detection = Detection.strategy_aware
            self.belief = self.update_belief()
            if self.belief[self.strategy_aware] == 0:
                self.detection = Detection.not_strategy_aware

    def _json(self):
        d = super()._json()
        d.pop("belief", None)
        d.pop("K", None)
        d.pop("K2", None)
        d.pop("learning", None)
        d.pop("arms", None)
        return d
