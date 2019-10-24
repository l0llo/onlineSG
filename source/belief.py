import numpy as np
from math import log, exp, sqrt
from copy import copy
import itertools
import source.util as util

class Belief:

    def __init__(self, profiles):
        self.profiles = profiles
        self.pr = {p: 1 / len(profiles) for p in self.profiles}

    def update(self, o):
        update = {p: p.last_strategy[o] * self.pr[p]
                  for p in self.profiles}
        norm = sum(update.values())
        self.v = {p: update[p] / norm for p in update}  # normalization


class BayesianBelief:

    def __init__(self, profiles):
        self.profiles = profiles
        self.alpha = {p: 1 for p in self.profiles}
        self.pr = {p: 1 / len(profiles) for p in self.profiles}

    def update(self, o):
        pr_given_t = {p.last_strategy[o] * self.pr[p]
                      for p in self.profiles}
        norm = sum([pr_given_t[p] for p in self.profiles])
        pr_given_t = {pr_given_t[p] / norm
                      for p in self.profiles}
        self.alpha = [self.alpha[p] + pr_given_t[p]
                      for p in self.profiles]
        belief_list = list(np.random.dirichlet([self.alpha[p]
                                                for p in self.profiles]))
        self.pr = {p: belief_list[i] for i, p in enumerate(self.profiles)}


class FrequentistBelief:

    def __init__(self, profiles, need_pr=False, ids=None):
        self.profiles = profiles
        if len(ids) == 1:
            self.loglk = {p: 0 for p in self.profiles}
            self.is_multi = False
        else:
            self.loglk = {id: {p: 0 for p in self.profiles if p.id == id} for id in ids}
            self.is_multi = True
        self.need_pr = need_pr
        if need_pr:
            self.pr = {p: 1 / len(profiles) for p in self.profiles}
        else:
            self.pr = None

    def hupdate(self, hdicts, history, ds_history, s_a):
        for p in self.profiles:
            hdicts[p]["last_strategy"] = s_a[p]
            self.loglk[p] = p.hloglk(self.loglk[p], hdicts[p],
                                     history, ds_history)
        t = len(self.profiles[0].game.strategy_history) + len(ds_history)
        if self.need_pr:
            self.compute_pr(t)

    def update(self):
        if not self.is_multi:
            for p in self.profiles:
                self.loglk[p] = p.loglk(self.loglk[p])
                if self.need_pr:
                    self.compute_pr(self.profiles[0].tau())
        else:
            for id in self.loglk.keys():
                for p in self.loglk[id].keys():
                    self.loglk[id][p] = p.loglk(self.loglk[id][p])

    def compute_pr(self, t):
        for p, x in self.loglk.items():
            if x is None:
                self.pr[p] = 0
            else:
                self.pr[p] = exp(x * t)

        norm = sum([self.pr[p] for p in self.profiles])
        if round(norm, 100) == 0:
            m = max([p for p in self.profiles if self.loglk[p] is not None],
                    key=lambda x: self.loglk[x])
            self.pr = {p: int(p is m) for p in self.profiles}
        else:
            self.pr = {p: self.pr[p] / norm for p in self.profiles}

    def get_copy(self):
        b = FrequentistBelief(self.profiles)
        b.loglk = copy(self.loglk)
        b.pr = copy(self.pr)
        return b

class MultiBelief:

    def __init__(self, profiles, num_att):
        self.profile_combinations = [tuple(x) for x in
                                     itertools.combinations_with_replacement(
                                     profiles, num_att)]
        self.loglk = {i: 0 for i in self.profile_combinations}

    def update(self, targets=None):
        for p in self.profile_combinations:
            self.loglk[p] = self.multiloglk(self.loglk[p], list(p), targets)

    def multiloglk(self, old_loglk, prof_list, targets):
        if old_loglk is None:
            return None
        lkl = 1
        for o in targets:
            lkl_o = 0
            for p in prof_list:
                prob_only_p_hits = p.last_strategy[o]
                for p1 in prof_list:
                    if p1 != p:
                        prob_only_p_hits *= 1 - p1.last_strategy[o]
                lkl_o += prob_only_p_hits
            lkl *= lkl_o
        if lkl == 0:
            return None
        new_l = log(lkl)
        return ((old_loglk * max(prof_list[0].tau() - 1, 0) + new_l) /
                max(prof_list[0].tau(), 1))

#class MultiProfileBelief:
#    def __init__(self, profiles):
#        self.profile_combinations = []
#        for i in self.profiles[0].game.attackers:
#            prof_of_i = []
#            for p in profiles:
#                if p.id == i:
#                    prof_of_i.append(p)
#            self.profile_combinations.append([tuple(x) for x in
#                                              itertools.combinations_with_replacement(
#                                              prof_of_i, len(prof_of_i))])
#        self.profile_combinations = list(itertools.product(*self.profile_combinations))
#        self.loglk = {i: [0] for i in self.profile_combinations}
#        self.freq = {i: [[0 for p in a] for a in i] for i in self.profile_combinations}

class MultiProfileBelief:
    def __init__(self, profiles):
        self.profiles = profiles
        num_prof_per_att = self.profiles[0].game.num_prof
        self.profile_combinations = list(itertools.combinations(profiles, num_prof_per_att))
        self.loglk = {i: 0 for i in self.profile_combinations}
        self.freq = {i: [0 for p in i] for i in self.profile_combinations}

    def update(self):
        o = self.profiles[0].game.history[-1][self.profiles[0].game.attackers[0]][0]
        dec_tau = max(len(self.profiles[0].game.history) - 1, 0)
        for profiles in list(self.loglk.keys()):
            max_b = -1
            max_p = None
            for i, p in enumerate(list(profiles)):
                if p.last_strategy[o] > max_b:
                    max_b = p.last_strategy[o]
                    max_p = i
            self.loglk[profiles] = (self.loglk[profiles] * dec_tau + log(max_b)) / max(dec_tau + 1, 1)
            self.freq[profiles][max_p] += 1
