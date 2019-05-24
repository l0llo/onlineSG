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
        self.has_game_observabilities = isinstance(self.game, gm.GameWithObservabilities)
        self.are_beliefs_all_zero = 1
        self.estimated_prob = dict()
#        self.threshold = 1

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles, need_pr=self.has_game_observabilities)
        for t in range(len(self.game.values)):
            self.estimated_prob[t] = [0, 0]

    def compute_strategy(self):
        valid_loglk = {p: self.belief.loglk[p] for p in self.A
                       if self.belief.loglk[p] is not None}
        if not self.has_game_observabilities:
            p = max (valid_loglk.keys(), key=lambda x: valid_loglk[x])
        else:
            if self.are_beliefs_all_zero:
#                valid_loglk = {p: self.belief.loglk[p] for p in self.A
#                               if self.belief.loglk[p] is not None}
#                max_loglk = max(valid_loglk.values())
#                max_keys = [k for k, v in valid_loglk.items() if v == max_loglk]
#                p = random.choice(max_keys)
                #in the beginning play uniform strategy (default for player class)
                return super().compute_strategy()
            else:
                valid_probs = {p: float(self.estimated_prob[p][0] /
                                        self.estimated_prob[p][1])
                               for p in self.estimated_prob.keys() if
                               self.estimated_prob[p][0] > 0}
                valid_keys = [int(k) for k in valid_probs.keys()]
                valid_probs_values = [float(x) for x in valid_probs.values()]
#                valid_probs_values.append(float(1 - sum(valid_probs_values)))
                print("estimated_prob is:", valid_probs_values)
                min_ent = sys.float_info.max
                min_ent_prof = None
                dist_len = len(valid_keys)
                for prof in self.game.profiles:
                    prof_strat = prof.compute_strategy()
                    valid_prof_strat = [prof_strat[t] for t in valid_keys]
#                    valid_prof_strat.append(float(1 - sum(valid_prof_strat)))
                    print("valid prof prob is:", valid_prof_strat)
#                    entropy = stats.entropy(valid_prof_strat,
#                                            valid_probs_values)

                    entropy = sum([valid_prof_strat[i] *
                                   log(valid_prof_strat[i]/valid_probs_values[i])
                                   for i in range(dist_len)])
                    if entropy < min_ent:
                        min_ent = entropy
                        min_ent_prof = prof
                p = min_ent_prof
#                diff = util.two_largest_diff(valid_loglk.values())
#                if diff >= self.threshold:
#                    p = max (valid_loglk.keys(), key=lambda x: valid_loglk[x])
#                else:
#                    valid_profiles = []
#                    beliefs = []
#                    for k,v in self.belief.pr.items():
#                        valid_profiles.append(k)
#                        beliefs.append(v)
#                    p = np.random.choice(valid_profiles, 1, p=beliefs)[0]
        return self.br_to(p)

    def learn(self):
        self.belief.update()
        #increase denominator
        self.estimated_prob[self.game.history[-1][0][0]][1] += 1
        if not self.game.fake_target[-1]:
            #increase numerator if observed
            self.estimated_prob[self.game.history[-1][0][0]][0] += 1
        if self.are_beliefs_all_zero:
            if any([p for p in self.belief.loglk.values()]):
                self.are_beliefs_all_zero = 0

        #extreme case where all profiles have been eliminated, make a clean slate of what was learnt
        if all([p is None for p in self.belief.loglk.values()]):
            self.belief.loglk = {p: 0 for p in self.A}
            self.are_beliefs_all_zero = 1
