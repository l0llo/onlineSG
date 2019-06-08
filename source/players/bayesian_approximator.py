import source.player as player
import source.standard_player_parsers as spp
from math import log
import re
import random as random
import numpy as np
from scipy.stats import dirichlet
import source.players.attackers as attackers


class BA(player.Defender):

    name = "BA"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.alpha = np.array([1 for t in range(len(self.game.values))])

    def compute_strategy(self):
#        sample = np.random.dirichlet(self.alphas, 1)
#        sample = tuple(dirichlet.rvs(self.alpha, size=1, random_state=1)[0].tolist())
        sample = dirichlet.rvs(self.alpha, size=1, random_state=1)[0].tolist()
        t = max([t for t,p in enumerate(sample)],
                key=lambda x: sample[x] * self.game.values[x][0])
        return self.ps(t)
#        return self.br_to(attackers.StochasticAttacker(self.game, 1, 1, *sample))
        #either compare with profiles' probability distributions or just draw from this distribution

    def learn(self):
        self.alpha[self.game.history[-1][1][0]] += 1
