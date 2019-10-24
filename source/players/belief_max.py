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
from operator import add
import scipy
import source.players.attackers as att
import itertools


class FB(player.Defender):

    name = "FB"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.belief = None
        self.learning_history = []
        self.use_lp = None
#        self.bound = 0
#        self.is_game_partial_feedback = isinstance(self.game, gm.PartialFeedbackGame)

    def finalize_init(self):
        super().finalize_init()
        if type(self.game).__name__ == "MultiProfileGame":
            self.belief = source.belief.MultiProfileBelief(self.game.profiles)
        elif len(self.game.attackers) > 1 and not self.game.dist_att:
            self.belief = source.belief.MultiBelief(self.game.profiles,
                                                    len(self.game.attackers))
        else:
            self.belief = source.belief.FrequentistBelief(self.game.profiles,
                                                          ids=self.game.attackers)#, need_pr=self.is_game_partial_feedback)
        self.use_lp = all([p.closed_form_sol for p in self.A])
#        self.compute_bound()

    def compute_strategy(self):
        if type(self.game).__name__ == "MultiProfileGame":
            return  self.compute_multi_prof_strategy()
        if len(self.game.attackers) > 1:
            return self.compute_multi_strategy()
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

    def compute_multi_strategy(self):
        if not self.game.dist_att:
            valid_loglk = {p: self.belief.loglk[p] for p in
                            self.belief.profile_combinations if
                            self.belief.loglk[p] is not None}
            max_belief_profiles = list(util.rand_max(valid_loglk.keys(),
                                                key=lambda x: valid_loglk[x]))
        else:
            max_belief_profiles = []
            for id in self.game.attackers:
                valid_loglk = {p: self.belief.loglk[id][p] for p in self.A
                                if (p.id == id and
                                self.belief.loglk[id][p] is not None)}
                p = util.rand_max(valid_loglk.keys(),
                                      key=lambda x: valid_loglk[x])
                max_belief_profiles.append(p)
        if self.use_lp:
            self.last_multi_br_to = self.multi_lp_br_to(max_belief_profiles)
        else:
            self.last_multi_br_to = self.multi_approx_br_to(max_belief_profiles)
        return self.last_multi_br_to

    def learn(self):
        if type(self.game).__name__ == "MultiProfileGame":
            self.belief.update()
        elif len(self.game.attackers) > 1 and not self.game.dist_att:
                self.belief.update([self.game.history[-1][a][0]
                                    for a in self.game.attackers])
        else:
            self.belief.update()#m)
            #extreme case where all profiles have been eliminated, make a clean slate of what was learnt
            if all([p is None for p in self.belief.loglk.values()]):
                self.belief.loglk = {p: 0 for p in self.A}

    def compute_multi_prof_strategy(self):
        valid_loglk = {p: self.belief.loglk[p] for p in self.belief.loglk.keys()
                       if self.belief.loglk[p] is not None}
        p = list(util.rand_max(valid_loglk.keys(), key=lambda x: valid_loglk[x]))
        self.learning_history.append('|'.join([*map(util.print_adv, p)]))
        if sum(self.belief.freq[tuple(p)]) == 0:
            prob = [1 / self.game.num_prof] * self.game.num_prof
        else:
            prob = [x / sum(self.belief.freq[tuple(p)]) for x in self.belief.freq[tuple(p)]]
#        prob = [x[1] for x in self.game.profile_distribution[0]]
        p_pr_l = (tuple([(i, j) for (i, j) in zip(p, prob)]),)
        sol = self.mp_br_to(p_pr_l)
        exp_loss = 0
        for p in self.game.profile_distribution[0]:
            exp_loss += p[1] * p[0].exp_loss({0: sol[:-1], 1: p[0].compute_strategy(sol[:-1])})
        self.last_exp_loss = exp_loss
        return sol[:-1]

    def compute_bound(self):
        best_responses = [self.br_to(p) for p in self.A]
        lambda_k_opt = self.compute_lambda_k(self.game.players[1], best_responses)
        a_str = None
        with io.StringIO() as buffer, redirect_stdout(buffer):
#            print(self.game.players[1])
            a_str = buffer.getvalue().replace("\n", "")
            buffer.close()
        for p in self.A:
            p_str = None
            with io.StringIO() as buffer, redirect_stdout(buffer):
#                print(p)
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
                if isinstance(self.game, gm.PartialFeedbackGame) \
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

#    def multi_br_to(self, profiles):
#        A_ub = []
#        for t in self.M:
#            terms = [self.game.values[t][self.id] * int(i != t)
#                     for i in self.M]
#            terms += [1]
#            terms += [0 for i in self.M]
#            A_ub.append(terms)
#        b_ub = [0 for i in range(len(A_ub))]
#        A_eq = [[1 for i in self.M] + [0] + [0 for i in self.M]]
#        b_eq = [[self.resources]]
#        for t in self.M:
#            A_eq.append([1 if (i == t or i == len(self.M) + t) else 0
#                         for i in range(2 * len(self.M) + 1)])
#            b_eq.append([1])
#        bounds = [(0, 1) for i in self.M] + [(None, None)] + [(0, 1) for i in self.M]
#        obj_fun = [0 for i in self.M] + [0] + [0 for i in self.M]
#        for p in profiles:
#            if isinstance(p, att.StackelbergAttacker):
#                obj_fun[len(self.M)] -= 1
#            else:
#                distr = p.compute_strategy()
#                for i, pr in enumerate(distr):
#                    obj_fun[len(self.M) + i + 1] -= pr * p.V[i]
#        print(A_eq, b_eq)
#        scipy_sol = list(scipy.optimize.linprog(obj_fun,
#                                                A_ub=np.array(A_ub),
#                                                b_ub=np.array(b_ub),
#                                                A_eq=np.array(A_eq),
#                                                b_eq=np.array(b_eq),
#                                                bounds=bounds,
#                                                method='simplex').x)
#        self.last_multi_br_to = scipy_sol[:-1]
#        return self.last_multi_br_to
