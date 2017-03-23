import source.player as player
import source.players.base_defenders as base_defenders
import source.standard_player_parsers as spp
import re
import source.errors as errors
import numpy as np
from math import exp
import source.util as util
from source.errors import NotFinalizedError


class StackelbergAttacker(player.Attacker):
    """
    The Stackelberg attacker observes the Defender strategy and plays a pure
    strategy that best responds to it.
    """

    name = "sta"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, g, id, resources=1):
        super().__init__(g, id, resources)
        self.br = None

    def compute_strategy(self):
        return self.best_respond(self.game.strategy_history[-1])

    def exp_loss(self, input_strategy):
        strategy = {0: input_strategy[0]}
        mock_attacker = player.Attacker(self.game, 1)
        att_strategy = mock_attacker.best_respond(strategy)
        strategy[1] = att_strategy
        return super().exp_loss(strategy)

    def opt_loss(self):
        sta_def = base_defenders.StackelbergDefender(self.game, 0, 1)
        sta_def.br_stackelberg()
        return -sta_def.maxmin

    def get_best_responder(self):
        if self.br is None:
            br = base_defenders.StackelbergDefender(self.game, 0)
            br.finalize_init()
            self.br = br
        return self.br


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
    The Dumb attacker, given an initially choosen action, always plays it
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

    name = "sto"
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

    def __init__(self, g, id, resources=1, *distribution):
        super().__init__(g, id, resources)
        if not distribution:
            self.distribution = util.gen_distr(len(g.values))
        else:
            self.distribution = list(distribution)

    def compute_strategy(self):
        return self.distribution

    def exp_loss(self, input_strategy):
        strategy = {0: input_strategy[0]}
        strategy[1] = self.distribution
        return super().exp_loss(strategy)

    def opt_loss(self):
        sto_def = base_defenders.KnownStochasticDefender(self.game, 0, 1, *
                                                         self.distribution)
        s = {0: sto_def.compute_strategy(),
             1: self.compute_strategy()}
        return self.exp_loss(s)

    def get_best_responder(self):
        br = base_defenders.KnownStochasticDefender(self.game, 0, 1,
                                                    *self.distribution)
        br.finalize_init()
        return br

    def __str__(self):
        return "-".join([super().__str__()] +
                        [str(d) for d in self.distribution])


class UnknownStochasticAttacker(player.Attacker):
    """
    Not a real attacker to be instantiated: it is intended to be used by the
    defender as a model
    """

    name = "unk_stochastic_attacker"
    pattern = re.compile(r"^" + name + r"\d$")

    def __init__(self, game, id, resources=1):
        super().__init__(game, id, resources)

    def compute_strategy(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        else:
            targets = list(range(len(self.game.values)))
            weights = {t: 0 for t in targets}
            for h in self.game.history:
                weights[h[self.id][0]] += 1
            norm = sum([weights[t] for t in targets])
            return [weights[t] / norm for t in targets]

    def get_best_responder(self):
        br = base_defenders.UnknownStochasticDefender2(self.game, 0,
                                                       mock_sto=self)
        br.finalize_init()
        return br

    def exp_loss(self, input_strategy):
        strategy = {0: input_strategy[0]}
        if self.last_strategy is None:
            self.play_strategy()
        strategy[1] = self.last_strategy
        return super().exp_loss(strategy)

    def opt_loss(self):
        if self.last_strategy is None:
            self.play_strategy()
        sto_def = base_defenders.KnownStochasticDefender(self.game, 0, 1, *
                                                         self.last_strategy)
        s = {0: sto_def.compute_strategy(),
             1: self.last_strategy}
        return self.exp_loss(s)


class SUQR(player.Attacker):

    name = "suqr"
    pattern = re.compile(r"^" + name + r"\d+(\.\d+)?(-\d+(\.\d+)?){3}$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, g, pl_id, L, w1, w2, c):
        super().__init__(g, pl_id, pl_id)
        self.L = L
        self.w1 = -w1
        self.w2 = w2
        self.c = c

    def compute_strategy(self):
        x = self.game.strategy_history[-1][0]
        return self.suqr_distr(x)

    def suqr_distr(self, x):
        targets = list(range(len(self.game.values)))
        R = [v[self.id] for v in self.game.values]
        q = np.array([exp(self.L * (self.w1 * x[t] +
                                    self.w2 * R[t] + 
                                    self.c))
                      for t in targets])
        q /= np.linalg.norm(q, ord=1)
        return list(q)

    def get_best_responder(self):
        br = base_defenders.SUQRDefender(self.game, 0, 1,
                                         mock_suqr=self)
        br.finalize_init()
        return br


    def exp_loss(self, input_strategy):
        strategy = {0: input_strategy[0]}
        R = [v[self.id] for v in self.game.values]
        att_strategy = self.suqr_distr(strategy[0])
        strategy[1] = att_strategy
        return super().exp_loss(strategy)

    def opt_loss(self):
        br = self.get_best_responder()
        return self.exp_loss({0: br.compute_strategy(),
                              1: None})
