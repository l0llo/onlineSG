# import source.players.base_defenders as bd
import source.standard_player_parsers as spp
import source.player as player
import source.belief
import re
import itertools
import source.util as util


class FR(player.Defender):
    """
    Hints On L-Moves at Each Step
    Good old backward induction on the next l-steps
    """
    name = "FR"
    pattern = re.compile(r"^" + name + r"\d+-\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources, h_max=1):
        super().__init__(game, id, resources)
        self.belief = None
        self.h_max = h_max
        # self.t_strategies = None
        # self.exploration = exploration

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles,
                                                      need_pr=True)

    def compute_strategy(self):
        R = self.reg_est(1, self.belief, [], [], {p: None for p in self.A})
        min_k = min(range(len(self.A)), key=lambda k: R[k])
        return self.br_to(self.A[min_k])

    def learn(self):
        self.belief.update()

    def reg_est(self, h, b, H, ds_history, hdicts):
        """
        implements the Regret Estimator recursive function (RE)
        It returns a list of s_node regrets.
        """

        r = dict()  # dict of the m_node regrets
        R = []  # list of the s_node regrets

        for k in self.A:
            s_d = self.br_to(k, hdict=hdicts[k])
            ds_history1 = ds_history + [s_d]
            for t in self.A:
                t.play_strategy(strategy=s_d, hdict=hdicts[t])
            s_a = {q: q.last_strategy for q in self.A}
            for i, j in X(self.M, self.M):
                H1 = H + [(i, j)]
                hdicts1 = {q: q.hlearn(H1, ds_history1, hdicts[q])
                           for q in self.A}
                b1 = b.get_copy()
                b1.hupdate(hdicts1, H1, ds_history1, s_a)
                if h < self.h_max:
                    R1 = self.reg_est(h + 1, b1, H1, ds_history1, hdicts1)
                    r[(i, j, k)] = min(R1)
                else:
                    r[(i, j, k)] = self.m_node_regret(H1, b1)
            R.append(self.s_node_regret(r, k, s_d, s_a, b))
        return R

    # problem with opt_loss FP -> it varies with time!
    # also US'one varies but it becomes more accurate

    def m_node_regret(self, H1, b1):
        return (sum([self.V[y] for x, y in H1 if x != y]) -
                self.h_max * sum([b1.pr[q] * q.opt_loss(history=H1)
                                  for q in self.A]))

    def s_node_regret(self, r, k, s_d, s_a, b):
        return sum([r[(x, y, k)] * s_d[x] * sum([b.pr[q] * s_a[q][y]
                                                 for q in self.A])
                    for x, y in X(self.M, self.M)])


def X(*args):
    """
    alias for the cartesian product function
    """
    return itertools.product(*args)

class FR1S(player.Defender):
    """
    1-step, no-unknown-profiles version of follow the regret
    """
    name = "FRL1S"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources):
        super().__init__(game, id, resources)
        self.belief = None
        # self.t_strategies = None
        # self.exploration = exploration

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.game.profiles,
                                                      need_pr=True)

    def compute_strategy(self):
        R = self.reg_est()
        min_k = min(range(len(self.A)), key=lambda k: R[k])
        return self.br_to(self.A[min_k])

    def learn(self):
        self.belief.update()

    def reg_est(self):
        R = []
        for k in self.A:
            R.append(0)
            s_d = self.br_to(k)
            ds_history = [s_d]
            for t in self.A:
                t.play_strategy(strategy=s_d)
            s_a = {q: q.last_strategy for q in self.A}
            for i, j in X(self.M, self.M):
                H = [(i, j)]
                hdicts = {q: q.hlearn(H, ds_history, None)
                           for q in self.A}
                b1 = self.belief.get_copy()
                b1.hupdate(hdicts, H, ds_history, s_a)
                R[-1] += 0 if i == j else ((self.V[j] - sum([b1.pr[q]
                                            * q.opt_loss(history=H1)
                                            for q in self.A])) * s_d[i]
                                            * sum([self.belief.pr[q] * s_a[q][j]
                                            for q in self.A]))
        return R

class FRL(player.Defender):
    """
    Lite version of follow the regret
    """

    name = "FRL"
    pattern = re.compile(r"^" + name + r"\d(-\d)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, pl_id, resources=1):
        super().__init__(game, pl_id, resources)
        self.belief = None
        self.exp_losses = dict()
#        self.is_game_partial_feedback = isinstance(self.game, gm.PartialFeedbackGame)

    def finalize_init(self):
        super().finalize_init()
        self.belief = source.belief.FrequentistBelief(self.A, need_pr=True)
        for p in self.A:
            self.exp_losses[p] = dict()
            for k in self.A:
                s_d = list(self.br_to(p))
                s_a = k.compute_strategy(s_d)
                self.exp_losses[p][k] = k.exp_loss({0:s_d,
                                                    1:s_a})

    def compute_strategy(self):
        R = self.reg_est()
        min_p = min(R.keys(), key=lambda p: R[p])
        return self.br_to(min_p)

    def reg_est(self):
        r = dict()
        for p in self.A:
            max_loss_k = util.rand_max(self.exp_losses[p].keys(),
                                       key=lambda x: self.exp_losses[p][x]
                                                     * self.belief.pr[x])
            r[p] = self.exp_losses[p][max_loss_k] * self.belief.pr[max_loss_k]
        return r

    def learn(self):
        self.belief.update()
