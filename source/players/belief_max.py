import source.player as player
import source.standard_player_parsers as spp
from math import log, exp
import re
import source.belief
import random as random
import copy as copy


class FB(player.Defender):

    name = "FB"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.belief = None

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles)

    def compute_strategy(self):
        valid_loglk = {p: self.belief.loglk[p] for p in self.A
                       if self.belief.loglk[p] is not None}
        #p = max (valid_loglk.keys(), key=lambda x: valid_loglk[x])
        max_loglk = max(valid_loglk.values())
        max_keys = [k for k, v in valid_loglk.items() if v == max_loglk]
        p = random.choice(max_keys)
        #id_prof = list(range(len(valid_loglk)))
        #keys_dict = {k: p for k, p in zip(id_prof, valid_loglk.keys())}
        #random.shuffle(id_prof)
        #k = max(id_prof, key=lambda x: valid_loglk[keys_dict[x]])
        #p = keys_dict[k]
        return self.br_to(p)

#    def learn(self):
#        self.belief.update()
#        #extreme case where all profiles have been eliminated, make a clean slate of what was learnt
#        if all([p is None for p in self.belief.loglk.values()]):
#            self.belief.loglk = {p: 0 for p in self.A}

    def learn(self):
        self.belief.update()
