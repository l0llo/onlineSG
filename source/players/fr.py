# import source.players.base_defenders as bd
import source.standard_player_parsers as spp
import source.player as player
import source.belief
import re
import itertools


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
        self.belief = source.belief.FrequentistBelief(self.game.profiles)

    def compute_strategy(self):
        R = self.reg_est(1, self.belief, [])
        min_k = min(range(len(self.A)), key=lambda k: R[k])
        return self.br_to(self.A[min_k])

    def learn(self):
        self.belief.update(self.game.history[-1][1][0])

    def reg_est(self, h, b, H):
        """
        implements the Regret Estimator recursive function (RE)
        It returns a list of s_node regrets.
        """

        r = dict()  # dict of the m_node regrets
        R = []  # list of the s_node regrets

        for k in self.game.profiles:
            s_d = self.br_to(k, history=H)
            for t in self.A:
                t.play_strategy(strategy=s_d, history=H)
            s_a = {q: q.last_strategy for q in self.A}
            for i, j in X(self.M, self.M):
                H1 = H + [(i, j)]
                b1 = b.get_copy()
                b1.update(j, add_time=len(H1))
                if h < self.h_max:
                    R1 = self.reg_est(h + 1, b1, H1)
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
