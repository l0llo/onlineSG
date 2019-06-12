import source.player as player
import source.standard_player_parsers as spp
from math import log, exp
import re
import source.belief
import random as random
import copy as copy
import source.game as gm
import numpy as np
import source.util as util
import scipy.stats as stats
import sys


class FB(player.Defender):

    name = "FB"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.belief = None
#        self.has_game_observabilities = isinstance(self.game, gm.GameWithObservabilities)

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles)#, need_pr=self.has_game_observabilities)

    def compute_strategy(self):
        valid_loglk = {p: self.belief.loglk[p] for p in self.A
                       if self.belief.loglk[p] is not None}
        p = util.rand_max(valid_loglk.keys(), key=lambda x: valid_loglk[x])

#       valid_profiles = []
#       beliefs = []
#       for k,v in self.belief.pr.items():
#           valid_profiles.append(k)
#           beliefs.append(v)
#       p = np.random.choice(valid_profiles, 1, p=beliefs)[0]
        return self.br_to(p)

    def learn(self):
        self.belief.update()
        #extreme case where all profiles have been eliminated, make a clean slate of what was learnt
        if all([p is None for p in self.belief.loglk.values()]):
            self.belief.loglk = {p: 0 for p in self.A}
