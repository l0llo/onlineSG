import source.player as player
import source.players.attackers as attackers
import source.standard_player_parsers as spp
from math import log, exp, sqrt
import scipy.stats as stats
import re
import source.belief
import sys
from scipy.stats import dirichlet
import numpy as np
import source.util as util

class DirichletMultinomialDefender(player.Defender):

    name = "DMD"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.dirichlet_alpha = {t: 0 for t in range(len(self.game.values))}
#        self.dirichlet_alpha = np.array([0 for t in range(len(self.game.values))])

    def compute_strategy(self):
        min_ent_prof = self.game.profiles[0]
        if not self.game.history:
            return self.br_to(min_ent_prof)
        #pk = list(self.dirichlet_alpha.values())
        #norm_pk = [float(i)/sum(pk) for i in pk]
        min_ent = sys.float_info.max
        non_zero = [i for i in range(len(self.game.values)) if self.dirichlet_alpha[i] != 0]
        alpha = np.array([self.dirichlet_alpha[i] for i in non_zero])
        norm_pk = dirichlet.rvs(alpha, size=1, random_state=1)[0].tolist()
        for p in self.game.profiles:
            p_str = p.compute_strategy()
            nz_p_str = [p_str[i] for i in non_zero]
#            divergences = [x * log(x / y) for x, y in zip (non_zero_pk, non_zero_p_str)]
#            entropy = abs(sum(divergences))
#            entropy = stats.entropy(nz_p_str, norm_pk)
            entropy = abs(sum([x * log(x / y) for x, y in zip(nz_p_str, norm_pk)]))
            if entropy < min_ent:
                min_ent = entropy
                min_ent_prof = p
#        self.curr_min_ent_prof = min_ent_prof
#        print("I'll best respond to", min_ent_prof)
        return self.br_to(min_ent_prof)
    ### MAYBE CHECK IF ANY NOT INFINITE FIRST WITHOUT EXCLUDING ZEROS, IF NONE, EXCLUDE ZEROS

    def learn(self):
#        if self.old_min_ent_prof and self.curr_min_ent_prof != self.old_min_ent_prof and isinstance(self.curr_min_ent_prof, attackers.StrategyAwareAttacker):
#            self.adjust_prior()
#        self.old_min_ent_prof = self.curr_min_ent_prof
        self.dirichlet_alpha[self.game.history[-1][1][0]] += 1

#    def adjust_prior(self):
#        prior_str = self.curr_min_ent_prof.compute_strategy()
#        game_length = len(self.game.history)
#        targets = range(len(self.game.values))
#        self.dirichlet_alphas = {t: prior_str[t] * game_length for t in targets}

class RE(player.Defender):

    name = "RE"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.actual_losses = None
        self.exp_losses = None
        self.last_prof = None

    def finalize_init(self):
        super().finalize_init()
        self.actual_losses = {p: [0, 0] for p in self.game.profiles}
        self.exp_losses = {p: {i: 0 for i in self.game.profiles} for p in self.game.profiles}
        self.last_prof = self.game.profiles[0]
        for p in self.exp_losses.keys():
            for i in self.game.profiles:
                def_strat = list(self.br_to(i))
                self.exp_losses[p][i] = p.exp_loss(def_strat)

    def compute_strategy(self):
        if self.tau() == 0:
            return self.br_to(self.last_prof)
        min_diff = sys.float_info.max
        for p in self.exp_losses.keys():
            diff_losses = 0
            for i in self.actual_losses.keys():
                if self.actual_losses[i][1] > 0:
                    diff_losses += abs(self.exp_losses[p][i]
                                       * self.actual_losses[i][1]
                                       - self.actual_losses[i][0]) ** 2
            print(p, "with diff loss", diff_losses)
            if diff_losses < min_diff:
                min_diff = diff_losses
                self.last_prof = p
        print("CHOSEN: ", self.last_prof)
        print("LOSSES: ", self.actual_losses)
        return self.br_to(self.last_prof)

    def learn(self):
        print("LAST PAYOFF: ", abs(sum(self.game.get_last_turn_payoffs(0))))
        self.actual_losses[self.last_prof][0] += abs(sum(self.game.get_last_turn_payoffs(self.id)))
        self.actual_losses[self.last_prof][1] += 1
