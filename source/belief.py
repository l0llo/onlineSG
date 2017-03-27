import numpy as np
from math import log, exp
from copy import copy


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

    def __init__(self, profiles):
        self.profiles = profiles
        self.loglk = {p: 0 for p in self.profiles}
        self.pr = {p: 1 / len(profiles) for p in self.profiles}

    def update(self, o):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        update = dict()
        for p in self.profiles:
            if p.last_strategy[o] == 0 or self.loglk[p] is None:
                update[p] = None
            else:
                update[p] = ((self.loglk[p] * (p.tau() - 1) +
                              log(p.last_strategy[o])) / p.tau())
        self.loglk = update

        for p, x in self.loglk.items():
            if x is None:
                self.pr[p] = 0
            else:
                self.pr[p] = exp(x * p.tau())

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
        b.pr = copy(self.loglk)
        return b
