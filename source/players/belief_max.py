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
import io
from contextlib import redirect_stdout


class FB(player.Defender):

    name = "FB"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.belief = None
#        self.bound = 0
#        self.has_game_observabilities = isinstance(self.game, gm.GameWithObservabilities)

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles)#, need_pr=self.has_game_observabilities)
#        self.compute_bound()

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
#        m = self.compute_most_informative_target()
        self.belief.update()#m)
        #extreme case where all profiles have been eliminated, make a clean slate of what was learnt
        if all([p is None for p in self.belief.loglk.values()]):
            self.belief.loglk = {p: 0 for p in self.A}

#    def compute_most_informative_target(self):
#        valid_loglk = {p: self.belief.loglk[p] for p in self.A
#                       if self.belief.loglk[p] is not None}
#        max_p = util.rand_max(valid_loglk.keys(), key=lambda x: valid_loglk[x])
#        valid_loglk.pop(max_p, None)
#        second_max_p = util.rand_max(valid_loglk.keys(), key=lambda x: valid_loglk[x])
#        profile_strategies = [p.compute_strategy(self.last_strategy)
#                              for p in self.A if self.belief.loglk[p] is not None]
#        profile_strategies = [p.compute_strategy(self.last_strategy)
#                              for p in [max_p, second_max_p]]
#        prob_by_target = [list(tup) for tup in zip(*profile_strategies)]
#        min_prob_diff = [util.find_min_diff(l, len(l)) for l in prob_by_target]
#        return np.argmax(min_prob_diff)

    def compute_bound(self):
        best_responses = [self.br_to(p) for p in self.A]
        lambda_k_opt = self.compute_lambda_k(self.game.players[1], best_responses)
        a_str = None
        with io.StringIO() as buffer, redirect_stdout(buffer):
            print(self.game.players[1])
            a_str = buffer.getvalue().replace("\n", "")
            buffer.close()
        for p in self.A:
            p_str = None
            with io.StringIO() as buffer, redirect_stdout(buffer):
                print(p)
                p_str = buffer.getvalue().replace("\n", "")
                buffer.close()
            if p_str != a_str:
                lambda_k = self.compute_lambda_k(p, best_responses)
                exp_regret_k = self.game.players[1].exp_loss(self.br_to(p))
                delta_b_k = self.compute_delta_b_k(self.game.players[1],
                                                  p, best_responses)
                self.bound += 2 * (lambda_k ** 2 + lambda_k_opt ** 2) * \
                              exp_regret_k / (delta_b_k ** 2)

    def compute_lambda_k(self, profile, br_list):
        min_belief = sys.float_info.max
        max_belief = sys.float_info.min
        for br in br_list:
            a_strat = profile.compute_strategy(br)
            for m in self.M:
                if a_strat[m] > max_belief:
                    max_belief = a_strat[m]
                elif a_strat[m] < min_belief and a_strat[m] != 0:
                    min_belief = a_strat[m]
                if isinstance(self.game, gm.GameWithObservabilities) \
                   and self.game.feedback_type == "mab":
                   if 1 - a_strat[m] > max_belief:
                       max_belief = 1 - a_strat[m]
                   elif 1 - a_strat[m] < min_belief:
                       min_belief = 1 - a_strat[m]
        return log(max_belief) - log(min_belief)

    def compute_delta_b_k(self, attacker, profile, br_list):
        min_diff = sys.float_info.max
        for br in br_list:
            a_strat = attacker.compute_strategy(br)
            p_strat = profile.compute_strategy(br)
            exp_a_belief = sum([a_strat[m] * a_strat[m] for m in self.M])
            exp_p_belief = sum([a_strat[m] * p_strat[m] for m in self.M])
            if exp_p_belief != 0 and exp_a_belief != 0:
                log_diff = log(exp_a_belief) - log(exp_p_belief)
                if log_diff < min_diff:
                    min_diff = log_diff
        return min_diff
