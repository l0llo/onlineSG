import source.player as player
import source.standard_player_parsers as spp
from math import exp, log, sqrt, pow
from copy import copy, deepcopy
from source.errors import AlreadyFinalizedError
import re
import numpy as np
import scipy.optimize
import source.util as util
import source.game as game

class PMF(player.Defender):

    name = "PMF"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources=1):
        super().__init__(game, id, resources)
        self.N = len(game.values)
        self.est_cum_loss = {e: 0 for e in range(self.N)}
        self.prob = {e: 1/self.N for e in range(self.N)}
        self.H = np.array([[1 if i != j else 0 for i in range(self.N)] for j in range(self.N)])
        self.K = None
        self.k_star = None
        self.eta = None
        self.gamma = None
        self.round = 1
        self.sel_arm = None

    def finalize_init(self):
            L = np.array([[self.game.values[j][0] if i != j else 0 for i in range(self.N)] for j in range(self.N)])
            H_inv = np.linalg.inv(self.H)
            self.K = np.ndarray.round(np.matmul(L, H_inv), 5)
            self.k_star = self.K.max()
            if self.k_star < 1:
                self.k_star = 1

    def compute_strategy(self):
        exp_distribution = self.compute_probabilities()
        self.sel_arm = range(self.N)[player.sample(exp_distribution, 1)[0]]
        return self.ps(self.sel_arm)

    def compute_probabilities(self):
        eta = pow(self.k_star, -2/3) * pow(log(self.N)/self.N, 2/3) \
              * pow(self.round, -2/3)
        gamma = pow(self.k_star, 2/3) * pow(self.N, 2/3) \
                * pow(log(self.N), 1/3) * pow(self.round, -1/3)
        prob = dict()
        for e in range(self.N):
            prob[e] = exp(-eta * self.est_cum_loss[e])
            if prob[e] < 0:
                prob[e] = 0
        norm = np.linalg.norm(np.array(list(prob.values())), ord=1)
        for e in range(self.N):
            self.prob[e] = (1 - gamma) * prob[e] / norm + gamma / self.N
        return [float(self.prob[e]) for e in range(self.N)]

    def learn(self):
        for e in range(self.N):
            est_loss = self.K[e][self.sel_arm] \
                       * self.H[self.sel_arm][self.game.history[-1][1][0]] \
                       / self.prob[self.sel_arm]
            self.est_cum_loss[e] += est_loss
        self.round += 1
