import source.players.base_defenders as base_defenders
import source.players.attackers as attackers
import source.standard_player_parsers as spp
import mpmath
from math import sqrt, log
import re


class FABULOUS(base_defenders.UnknownStochasticDefender2):

    name = "fabulous"
    pattern = re.compile(r"^" + name + r"\d+(-\d+(\.\d+)?)$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, g, index, resources=1, power=2):
        super().__init__(g, index, resources)
        self.profiles = []
        self.unk_included = True
        self.sta_att = None
        self.avg_loss = 0
        self.exp_adv_loss = None
        self.power = power

    def finalize_init(self):
        super().finalize_init()
        self.profiles = self.game.get_profiles_copies()
        unknowns = [p for p in self.profiles
                    if isinstance(p, attackers.UnknownStochasticAttacker)]
        stackelbergs = [p for p in self.profiles
                        if isinstance(p, attackers.StackelbergAttacker)]
        if len(stackelbergs) > 0:
            self.sta_att = stackelbergs[0]
        for u in unknowns:
            self.profiles.remove(u)
        self.exp_adv_loss = {p: 0 for p in self.profiles}

    def compute_strategy(self):
        if self.tau() > 1:
            p = 0.5 / (self.tau() ** self.power)
            b = (self.norm_const *
                 sqrt(-log(p) / (self.tau())))
            interval = mpmath.mpi(self.avg_loss - b,
                                  self.avg_loss + b)
            for p in sorted(self.profiles, key=lambda x: self.exp_adv_loss[x]):
                if self.exp_adv_loss[p] in interval:
                    print("exp_loss", self.exp_adv_loss[p], "is in", interval)
                    print("with b", b, "and avg_loss", self.avg_loss)
                    return p.get_best_responder().compute_strategy()
        print("unknown")
        # MODIFY to take into account the not-Unknown case
        # if I have not returned yet, then I must use fpl
        return super().compute_strategy()

    def learn(self):
        super().learn()
        for p in self.profiles:
            loss_if_p = p.exp_loss({0: self.game.strategy_history[-1][self.id],
                                    1: None})
            self.exp_adv_loss[p] = ((self.exp_adv_loss[p] * (self.tau() - 1) +
                                     loss_if_p) / self.tau())
            self.avg_loss = ((self.avg_loss * (self.tau() - 1) -
                              self.feedbacks[-1]["total"]) / self.tau())

# LossTuple = namedtuple('LossTuple', ['loss', 'br', 'attacker'])


# class FABULOUS(base_defenders.UnknownStochasticDefender2):

#     name = "fabulous"
#     pattern = re.compile(r"^" + name + r"\d+$")

#     @classmethod
#     def parse(cls, player_type, game, id):
#         return spp.base_parse(cls, player_type, game, id)

#     def __init__(self, g, index, resources=1):
#         super().__init__(g, index, resources)
#         self.opt_lt = []
#         self.profiles = []
#         self.unk_included = True
#         self.sta_att = None

#     def finalize_init(self):
#         super().finalize_init()
#         self.profiles = self.game.get_profiles_copies()
#         self.opt_lt = sorted([LossTuple(p.opt_loss(),
#                                         p.get_best_responder()
#                                         .compute_strategy(),
#                                         p)
#                               for p in self.profiles
#                               if isinstance(p, attackers.StochasticAttacker)],
#                              key=lambda x: x.loss)
#         # probably it has to be changed
#         self.unk_included = bool(len([p for p in self.profiles
#                                       if isinstance(p,
#                                                     attackers.
#                                                     StochasticAttacker)]))
#         stackelbergs = [p for p in self.profiles
#                         if isinstance(p,
#                                       attackers.
#                                       StackelbergAttacker)]
#         if len(stackelbergs) > 0:
#             self.sta_att = stackelbergs[0]

#     def compute_strategy(self):
#         if self.tau() > 1:
#             p = 0.5 / (self.tau() ** 2)
#             b = (self.norm_const *
#                  sqrt(-log(p) / (self.tau())))
#             # if self.sta_att is not None:
#             #     exp_unk_loss = self.mock_sto.exp_loss({0: self.emp_distr(),
#             #                                            1: None})
#             #     avg_loss = (-sum([f['total'] for f in self.feedbacks]) /
#             #                 self.tau())
#             #     interval = mpmath.mpi(avg_loss - b, avg_loss + b)
#             #     if exp_unk_loss not in interval:
#             #         print("exp_loss", exp_unk_loss, "is in", interval)
#             #         print("with b", b, "and avg_loss", avg_loss)
#             #         return self.sta_att.get_best_responder().compute_strategy()
#             for lt in self.opt_lt:
#                 loss = [-self.avg_rewards[k] for k
#                         in self.avg_rewards
#                         if k.fixed_strategy == lt.br][0]
#                 interval = mpmath.mpi(loss - b, loss + b)
#                 if lt.loss in interval:
#                     print("lt_loss", lt.loss, "is in", interval)
#                     print("with b", b, "and loss", loss)
#                     return lt.br
#             print("unknown")
#         # MODIFY to take into account the not-Unknown case
#         # if I have not returned yet, then I must use fpl
#         return super().compute_strategy()

#     def emp_distr(self):
#         targets = list(range(len(self.game.values)))
#         weights = {t: 0 for t in targets}
#         for h in self.game.history:
#             weights[h[self.id][0]] += 1
#         norm = sum([weights[t] for t in targets])
#         return [weights[t] / norm for t in targets]
