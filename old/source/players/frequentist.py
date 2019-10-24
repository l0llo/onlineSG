import source.player as player
import source.players.attackers as attackers
import source.players.belief_max as fb
import source.standard_player_parsers as spp
from math import log, exp
import scipy.stats as stats
import re
import source.belief
import sys
import source.errors as errors

"""
THIS CLASS IS OBSOLETE
"""

class FD(fb.FB):

    name = "FD"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
#        self.profiles = None
        targets = range(len(game.values))
#        self.dirichlet_alphas = {t: 0 for t in targets}
        self.epsilon = 0.5 # set this threshold to the tolerated belief/entropy
        self.unknown_profile = []
        self.known_profs = []
        if len(self.unknown_profile) > 1:
            raise errors.MultipleUnknownProfilesError()

    def finalize_init(self):
        player.Player.finalize_init(self)
        for p in self.game.profiles:
            if isinstance(p, attackers.FrequentistUnknownStochasticAttacker):
                self.unknown_profile.append(p)
            else:
                self.known_profs.append(p)
#        known_prof = [p for p in self.game.profiles if not isinstance(p, attackers.FrequentistUnknownStochasticAttacker)]
        self.belief = source.belief.FrequentistBelief(self.known_profs)

#    def finalize_init(self):
#        super().finalize_init()
#        self.profiles = self.game.profiles

#    def compute_strategy(self):
#        if not self.game.strategy_history:
#            return self.br_to(attackers.StackelbergAttacker(self.game, 1, 1))
#        pk = list(self.dirichlet_alphas.values())
#        norm_pk = [float(i)/sum(pk) for i in pk]
#        unknown_params_profile = None
#        min_ent = sys.float_info.max
#        min_ent_prof = self.profiles[0]
#        for p in self.profiles:
#            if isinstance(p, attackers.FrequentistUnknownStochasticAttacker):
#                unknown_params_profile = p
#                p.set_weights(norm_pk)
#            else:
#                p_str = p.compute_strategy()
#                entropy = stats.entropy(norm_pk, p_str)
#                if entropy < min_ent:
#                    min_ent = entropy
#                    min_ent_prof = p
#        if min_ent < self.epsilon or not unknown_params_profile:
#            return self.br_to(min_ent_prof)
#        else:
#            unknown_params_profile.set_weights(norm_pk)
#            return self.br_to(unknown_params_profile)

    def compute_strategy(self):
        valid_loglk = {p: self.belief.loglk[p] for p in self.known_profs
                       if self.belief.loglk[p] is not None}
        if valid_loglk:
            p = max(valid_loglk.keys(), key=lambda x: valid_loglk[x])
            if valid_loglk[p] >= self.epsilon:
                return self.br_to(p)
        return self.br_to(self.unknown_profile[0])

    def learn(self):
        self.belief.update()
        o = self.game.history[-1][1][0]
        self.unknown_profile[0].set_weights(o)
#        self.dirichlet_alphas[o] += 1
#        pk = list(self.dirichlet_alphas.values())
#        norm_pk = [float(i)/sum(pk) for i in pk]
#        self.unknown_profile[0].set_weights(norm_pk)
