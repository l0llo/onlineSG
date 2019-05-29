import source.player as player
import source.standard_player_parsers as spp
from math import log
import re
import random as random
import numpy as np
import source.util as util
import scipy.stats as stats
import sys


class BA(player.Defender):

    name = "BA"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.observed_something = 0
        self.estimated_prob = dict()

    def finalize_init(self):
        super().finalize_init()
        for t in range(len(self.game.values)):
            self.estimated_prob[t] = [0, 0]

    def compute_strategy(self):
        if not self.observed_something:
            #in the beginning play uniform strategy (default for player class)
            return super().compute_strategy()
        else:
            valid_probs = {p: float(self.estimated_prob[p][0] /
                                    self.estimated_prob[p][1])
                           for p in self.estimated_prob.keys() if
                           self.estimated_prob[p][0] > 0}
            valid_keys = [int(k) for k in valid_probs.keys()]
#            valid_probs_values = [float(x) for x in valid_probs.values()]
            min_ent = sys.float_info.max
            min_ent_prof = None
            for prof in self.game.profiles:
                prof_strat = prof.compute_strategy()
#                valid_prof_strat = [prof_strat[t] for t in valid_keys]
                entropy = sum([prof_strat[k] *
                               log(prof_strat[k] / valid_probs[k])
                               for k in valid_keys])
                if entropy < min_ent:
                    min_ent = entropy
                    min_ent_prof = prof
            p = min_ent_prof
        return self.br_to(p)

    def learn(self):
        #increase denominator
        self.estimated_prob[self.game.history[-1][0][0]][1] += 1
        if not self.game.fake_target[-1]:
            #increase numerator if observed
            self.estimated_prob[self.game.history[-1][0][0]][0] += 1
            if not self.observed_something:
                self.observed_something = 1
