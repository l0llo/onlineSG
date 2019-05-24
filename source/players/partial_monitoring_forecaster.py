import source.player as player
import source.standard_player_parsers as spp
from math import exp, log, sqrt
from copy import copy, deepcopy
from source.errors import AlreadyFinalizedError
import re
import numpy as np
import scipy.optimize
import source.util as util
import source.game as game

class PMForecaster(player.Defender):

    name = "PMF"
      pattern = re.compile(r"^" + name + r"\d$")

      @classmethod
      def parse(cls, player_type, game, id):
          return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources):
        super().__init__(game, id, resources, 1)
        self.est_cum_losses = {e: 0 for e in game.values}
        self.prob = {e: 0 for e in game.values}
        self.k_star = None
        self.eta = None
        self.gamma = None
        self.K = None
        self.H = None
        self.N = len(self.game.values)
        self.round = 0

    def finalize_init(self):
            L = matrix([[self.game.values[j][0] if i != j else 0 for i in self.game.values] for j in self.game.values])
            self.H = matrix([[1 if i != j else 0 for i in self.game.values] for j in self.game.values])
            H_inv = H.I
            self.K = L * H_inv
            self.k_star = K.max()
            if self.k_star < 1:
                self.k_star = 1

    def learn(self):
        for e in range(self.N):
            est_loss = self.K[e][self.sel_arm] * self.H[self.sel_arm][self.game.history[-1][1][0]] \
                       / self.prob[e]
            self.est_cum_losses[e] = self.est_cum_losses[e] + est_loss
        self.round += 1

    def compute_strategy(self):
        exp_distribution = self.compute_probabilities()
        self.sel_arm = self.arms[player.sample(exp_distribution, 1)[0]]
        return self.sel_arm.play_strategy()

    def compute_probabilities(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.arms))
        eta = self.k_star^(-2/3) * (log(self.N)/self.N)^(2/3) * self.round^(-2/3)
        gamma = self.k_star^(2/3) * self.N^(2/3) * (log(self.N))^(1/3) * self.round^(-1/3)
        prob = dict()
        for e in range(self.N):
            prob[e] = exp(-eta * self.est_cum_losses[e])
        norm = np.linalg.norm(np.array(list(prob_tilde.values())), ord=1)
        for e in range(self.N):
            prob[e] = (1 - gamma) * prob[e] / norm + gamma / self.N
        return [float(self.prob[e]) for e in range(self.N)]
