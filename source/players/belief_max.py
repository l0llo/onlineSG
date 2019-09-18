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
#        self.bound = 0
#        self.is_game_partial_feedback = isinstance(self.game, gm.PartialFeedbackGame)

    def finalize_init(self):
        super().finalize_init()
        if len(self.game.attackers) > 1 and not self.game.dist_att:
            self.belief = source.belief.MultiBelief(self.game.profiles,
                                                    len(self.game.attackers))
        else:
            self.belief = source.belief.FrequentistBelief(self.game.profiles)#, need_pr=self.is_game_partial_feedback)
#        self.compute_bound()

    def compute_strategy(self):
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
            for pl in self.game.players.values():
                id = pl.id
                if id in self.game.attackers:
                    valid_loglk = {p: self.belief.loglk[p] for p in self.A
                                    if (self.belief.loglk[p] is not None
                                    and p.id == id)}
                    p = util.rand_max(valid_loglk.keys(),
                                      key=lambda x: valid_loglk[x])
                    max_belief_profiles.append(p)
        return self.multi_br_to(max_belief_profiles)

    def learn(self):
#        m = self.compute_most_informative_target()
        if len(self.game.attackers) > 1 and not self.game.dist_att:
            self.belief.update([self.game.history[-1][p]
                                for p in self.game.players if p != self.id])
        else:
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

    def multi_br_to(self, profiles):
        def fun(x):
            return sum(p.exp_loss(x) for p in profiles)
        bnds = tuple([(0, 1) for t in self.M])
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        res = scipy.optimize.minimize(fun, util.gen_distr(len(self.M)),
                                      method='SLSQP', bounds=bnds,
                                      constraints=cons, tol=0.000001)
        self.last_multi_br_to = list(res.x)
        return self.last_multi_br_to

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
