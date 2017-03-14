import source.players.base_defenders as base_defenders
import source.players.attackers as attackers
import source.standard_player_parsers as spp
import mpmath
from collections import namedtuple
from math import sqrt, log
import re
LossTuple = namedtuple('LossTuple', ['loss', 'br', 'attacker'])


class FABULOUS(base_defenders.UnknownStochasticDefender2):

    name = "fabulous"
    pattern = re.compile(r"^" + name + r"\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    def __init__(self, g, index, resources=1):
        super().__init__(g, index, resources)
        self.opt_lt = []
        self.profiles = []
        self.unk_included = True

    def finalize_init(self):
        super().finalize_init()
        self.profiles = self.game.get_profiles_copies()
        self.opt_lt = sorted([LossTuple(p.opt_loss(),
                                        p.get_best_responder()
                                        .compute_strategy(),
                                        p)
                              for p in self.profiles
                              if isinstance(p, attackers.StochasticAttacker)],
                             key=lambda x: x.loss)
        # probably it has to be changed
        self.unk_included = bool(len([p for p in self.profiles
                                      if isinstance(p,
                                                    base_defenders.
                                                    UnknownStochasticDefender2)]))

    def compute_strategy(self):
        if self.tau() > 1:
            for lt in self.opt_lt:
                loss = [-self.avg_rewards[k] for k
                        in self.avg_rewards
                        if k.fixed_strategy == lt.br][0]
                p = 0.5 / (self.tau() ** 2)
                b = (self.norm_const *
                     sqrt(-log(p) / (self.tau())))
                interval = mpmath.mpi(loss - b, loss + b)
                if lt.loss in interval:
                    # print("lt_loss", lt.loss, "is in", interval)
                    # print("with b", b, "and loss", loss)
                    return lt.br
        # MODIFY to take into account the not-Unknown case
        # if I have not returned yet, then I must use fpl
        return super().compute_strategy()
