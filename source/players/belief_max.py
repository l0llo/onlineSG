import source.player as player
import source.standard_player_parsers as spp
from math import log, exp
import re
import source.belief


class FB(player.Defender):

    name = "FB"
    pattern = re.compile(r"^" + name + r"\d$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources):
        super().__init__(game, pl_id, resources)
        self.belief = None

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles)

    def compute_strategy(self):
        p = max(self.game.profiles, key=lambda x: self.belief.pr[x])
        return self.br_to(p)

    def learn(self):
        self.belief.update(self.game.history[-1][1][0])
