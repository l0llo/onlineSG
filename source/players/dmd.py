import source.player as player
import source.players.attackers as attackers
import source.standard_player_parsers as spp
from math import log, exp
import scipy.stats as stats
import re
import source.belief
import sys

class DirichletMultinomialDefender(player.Defender):

    name = "DMD"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.profiles = None
        targets = range(len(self.game.values))
        self.dirichlet_alphas = {t: 0 for t in targets}
#        self.old_min_ent_prof = None
#        self.curr_min_ent_prof = None

    def finalize_init(self):
        super().finalize_init()
        self.profiles = self.game.profiles

    def compute_strategy(self):
        pk = list(self.dirichlet_alphas.values())
        norm_pk = [float(i)/sum(pk) for i in pk]
#        print("Approximate distribution is:", norm_pk)
        if not self.game.strategy_history:
            return self.br_uniform() # first turn apply uniform strategy
        min_ent = sys.float_info.max
        min_ent_prof = self.profiles[0]
        for p in self.profiles:
            p_str = p.compute_strategy()
#            non_zero_prob = [i for i in range(len(p_str)) if p_str[i] != 0 and norm_pk[i] != 0]
#            non_zero_prob_pk = [norm_pk[i] for i in non_zero_prob]
#            non_zero_prob_p_str = [p_str[i] for i in non_zero_prob]
#            print("Comparing non zero distr", non_zero_prob_pk, " and", non_zero_prob_p_str)
#            divergences = [x * log(x / y) for x, y in zip (non_zero_prob_pk, non_zero_prob_p_str)]
#            entropy = abs(sum(divergences))
            entropy = stats.entropy(norm_pk, p_str)
#            print("Profile", p, " with strat", p_str, " has ent", entropy)
            if entropy < min_ent:
                min_ent = entropy
                min_ent_prof = p
#        self.curr_min_ent_prof = min_ent_prof
#        print("I'll best respond to", min_ent_prof)
        return self.br_to(min_ent_prof)

    def learn(self):
#        if self.old_min_ent_prof and self.curr_min_ent_prof != self.old_min_ent_prof and isinstance(self.curr_min_ent_prof, attackers.StrategyAwareAttacker):
#            self.adjust_prior()
#        self.old_min_ent_prof = self.curr_min_ent_prof
        o = self.game.history[-1][1][0]
        self.dirichlet_alphas[o] += 1

    def adjust_prior(self):
        prior_str = self.curr_min_ent_prof.compute_strategy()
        game_length = len(self.game.history)
        targets = range(len(self.game.values))
        self.dirichlet_alphas = {t: prior_str[t] * game_length for t in targets}
