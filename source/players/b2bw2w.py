import source.player as player
import source.standard_player_parsers as spp
from math import log, exp
import re
import numpy as np


class B2BW2W(player.Defender):
    """
    "Bread to bread, wine to wine"
    """

    name = "b2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, game, id, resources):
        super().__init__(game, id, resources)
        self.profiles = None
        self.belief = None
        self.arms = None
        self.sel_arm = None
        self.learning = player.Learning.EXPERT

    def finalize_init(self):
        super().finalize_init()
        self.profiles = self.game.get_profiles_copies()
        for p in self.profiles:
            p.finalize_init()
        self.belief = {k: 1 / (len(self.profiles)) for k in self.profiles}
        self.arms = {k: k.get_best_responder() for k in self.profiles}

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = {k: k.last_strategy[o] * self.belief[k]
                  for k in self.profiles}
        eta = 1 / sum(update.values())
        update = {k: update[k] * eta for k in update}  # normalization
        return update

    def compute_strategy(self):
        chosen = player.sample([self.belief[k] for k in self.profiles], 1)[0]
        self.sel_arm = self.arms[self.profiles[chosen]]
        return self.sel_arm.play_strategy()

    def learn(self):
        """ make our imagined adversary make a move
            pay attention to history-based ones!!!
            (use self.tau() somehow?) """
        for k in self.profiles:
            k.play_strategy()
        self.belief = self.update_belief()

    def _json(self):
        d = super()._json()
        d.pop("arms", None)
        d.pop("belief", None)
        d.pop("profiles", None)
        d.pop("learning", None)
        return d


class MB2BW2W(B2BW2W):

    name = "mb2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    def compute_strategy(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        chosen = max(self.profiles, key=lambda x: self.belief[x])
        self.sel_arm = self.arms[chosen]
        return self.sel_arm.play_strategy()


class FB2BW2W(B2BW2W):

    name = "fb2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    def finalize_init(self):
        super().finalize_init()
        self.belief = {k: 0 for k in self.profiles}

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = dict()
        for p in self.profiles:
            if p.last_strategy[o] == 0 or self.belief[p] is None:
                update[p] = None
            else:
                update[p] = ((self.belief[p] * (self.tau() - 1) +
                              log(p.last_strategy[o])) / self.tau())
        return update

    def compute_strategy(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        norm_belief = [self.belief[p] for p in self.profiles]
        for i, b in enumerate(norm_belief):
            if b is None:
                norm_belief[i] = 0
            else:
                norm_belief[i] = exp(b * self.tau())

        norm_belief = np.array(norm_belief)
        norm = np.linalg.norm(norm_belief, ord=1)
        if round(norm, 100) == 0:
            chosen = max(self.profiles, key=lambda x: self.belief[x])
            self.sel_arm = self.arms[chosen]
            return self.sel_arm.play_strategy()
        norm_belief /= norm
        chosen = player.sample(list(norm_belief), 1)[0]
        self.sel_arm = self.arms[self.profiles[chosen]]
        return self.sel_arm.play_strategy()


class MFB2BW2W(FB2BW2W):

    name = "mfb2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    def compute_strategy(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        chosen = max([p for p in self.profiles if self.belief[p] is not None],
                     key=lambda x: self.belief[x])
        self.sel_arm = self.arms[chosen]
        return self.sel_arm.play_strategy()


class BB2BW2W(B2BW2W):

    name = "bb2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    def __init__(self, game, pl_id, resources):
        super().__init__(game, pl_id, resources)
        self.alpha = None

    def finalize_init(self):
        super().finalize_init()
        self.alpha = [1 for k in self.profiles]

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        pk_given_t = np.array([k.last_strategy[o] * self.belief[k]
                               for k in self.profiles])
        pk_given_t /= np.linalg.norm(pk_given_t, ord=1)
        pk_given_t = list(pk_given_t)
        self.alpha = [self.alpha[i] + pk_given_t[i]
                      for i, k in enumerate(self.profiles)]
        belief_list = list(np.random.dirichlet(self.alpha))
        update = {k: belief_list[i] for i, k in enumerate(self.profiles)}
        return update


class MBB2BW2W(BB2BW2W):

    name = "mbb2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")

    def compute_strategy(self):
        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        chosen = max(self.profiles, key=lambda x: self.belief[x])
        self.sel_arm = self.arms[chosen]
        return self.sel_arm.play_strategy()
