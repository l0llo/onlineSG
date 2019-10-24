"""
ROADMAP

1 defender and 1 attacker, with one resource each

- Possible types:
    - Known Stochastic      V
    - Unknown Stochastic    V
    - Stackelberg           V
    - Fictitious Player     ? (stackelberg?)
    - Quantal Response      -
    - Adversary             -

- 2 types:
    -1 Known Stochastic or Stackelberg           V
    -2 Unknown Stochastic or Stackelberg         ~
    -3 Known Stochastic or Fictitious Player     ?
    -4 Unknown Stochastic or Fictitious Player   -
    -5 Stackelberg or Fictitious Player          ?
    -6 Known Stochastic or Quantal Response      -
    -7 Unknown Stochastic or Quantal Response    -
    -8 Stackelberg or Quantal Response           -
    -9 Fictitious Player or Quantal response     -
    -10 Known Stochastic or Adversary            -
    -11 Unknown Stochastic or Adversary          -
    -12 Stackelberg or Adversary                 -
    -13 Fictitious Player or Adversary           -
    -14 Quantal Response or Adversary            -

Then we can try to distinguish among sets of types:

- discriminate among a set by eliminating one type at time using the known
  bounds
- e.g. {KS, Sta, QR}: 2 bounds for KS, see which is passed before
- if it is passed => discard KS!
- otherwise => it could be KS


Extend the 2-types identification with multiple resources

Extend the multiple-types

Extend the number of attackers (not coordinated)

Extend the number of defenders


"""

import source.player as player
import source.players.attackers as attackers
import source.players.base_defenders as base_defenders
import source.standard_player_parsers as spp
from math import log, sqrt
from copy import deepcopy
import enum
import re

State = enum.Enum('State', 'exploring stochastic stackelberg')


# class FABULOUS(base_defenders.StackelbergDefender,
#                base_defenders.KnownStochasticDefender):
#     """
#     Adapt When Everybody is Stochastic, Otherwise Move to Stackelberg
#     with Upper Confidence Bound where p=t^(-4)

#     Defender detection of Stackelberg-Known Stochastic
#     """

#     name = "fabulous"
#     pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

#     @classmethod
#     def parse(cls, player_type, game, id):
#         return spp.stochastic_parse(cls, player_type, game, id)

#     def __init__(self, game, id, resources=1, *distribution):
#         player.Defender.__init__(self, game, id, resources)
#         self.distribution = distribution
#         self.maxmin = None
#         self.stochastic_reward = None
#         self.br_stackelberg_strategy = None
#         self.br_stochastic_strategy = None

#         # Initialization
#         self.norm_const = 1  # has to be initialized late
#         self.br_stackelberg()
#         self.br_stochastic()
#         self.stochastic_loss = - self.stochastic_reward
#         self.loss_sta_sto = None
#         self.state = State.exploring

#     def compute_strategy(self):
#         # if it is the first round then br to stackelberg
#         t = len(self.game.history)
#         if t == 0:
#             mock_stackelberg = attackers.StackelbergAttacker(self.game, 1)
#             strategies = {0: self.br_stochastic(),
#                           1: None}
#             self.norm_const = (mock_stackelberg.exp_loss(strategies) -
#                                self.stochastic_loss)
#         if t < 2:
#             return self.br_stochastic()
#         else:
#             def_last_moves = set(self.game.history[-1][0])  # hardcoded
#             att_last_moves = set(self.game.history[-1][1])
#             if self.state == State.exploring:
#                 if def_last_moves.intersection(att_last_moves):
#                     self.state = State.stochastic
#                     return self.br_stochastic()
#                 avg_loss = - sum([f['total'] for f in self.feedbacks]) / t
#                 delta_t = 1 / (t * t)
#                 bound = (self.norm_const *
#                          sqrt(-log(delta_t) / (t)))
#                 if avg_loss - self.stochastic_loss <= bound:
#                     return self.br_stochastic()
#                 else:
#                     # print("switch at: ", t)
#                     self.state = State.stackelberg
#                     return self.br_stackelberg()
#             elif self.state == State.stochastic:
#                 return self.br_stochastic()
#             elif self.state == State.stackelberg:
#                 return self.br_stackelberg()

#     def _json(self):
#         self_copy = deepcopy(self)
#         d = self_copy.__dict__
#         d.pop("game", None)
#         d["state"] = d["state"].name
#         d["class_name"] = self.__class__.__name__
#         return d


class BR_Expert(base_defenders.ExpertDefender):

    name = "br_expert"
    pattern = re.compile(r"^" + name + r"\d+-\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources, algo='fpl'):
        experts = []
        super().__init__(game, id, resources, algo, *experts)

    def finalize_init(self):
        profiles = self.game.get_profiles_copies()
        for p in profiles:
            p.finalize_init()
        self.arms = [p.get_best_responder() for p in profiles]
        super().finalize_init()

    def __str__(self):
        return "-".join([super().__str__()] + ["1"])


class BR_MAB(base_defenders.MABDefender):

    name = "br_mab"
    pattern = re.compile(r"^" + name + r"\d+-\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources, algo='fpl'):
        experts = []
        super().__init__(game, id, resources, algo, *experts)

    def finalize_init(self):
        profiles = self.game.get_profiles_copies()
        for p in profiles:
            p.finalize_init()
        self.arms = [p.get_best_responder() for p in profiles]
        super().finalize_init()

    def __str__(self):
        return "-".join([super().__str__()] + ["1"])
