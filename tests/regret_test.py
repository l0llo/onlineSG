import unittest
import logging
import source.util as util
import source.game as game
import source.players.attackers as atk
import source.runner as runner
import source.players.belief_max as bm
from copy import deepcopy


logger = logging.getLogger(__name__)


class FB_RegretTestCase(unittest.TestCase):
    """
    verify that the algorithm has a constant regret, namely we check that the
    last 10 values are equal.
    """

    def setUp(self):
        time_horizon = 100
        targets = util.gen_norm_targets(10)
        self.game = game.zs_game(targets, time_horizon)
        self.profiles = util.gen_profiles(targets,
                                          [(atk.UnknownStochasticAttacker, 1),
                                           (atk.StackelbergAttacker, 1),
                                           (atk.SUQR, 1),
                                           (atk.StochasticAttacker, 1),
                                           (atk.FictitiousPlayerAttacker, 1)])

    def test_stochastic(self):
        self.const_regret(3)

    def test_stackelberg(self):
        self.const_regret(1)

    def test_suqr(self):
        self.const_regret(2)

    def test_unk_sto(self):
        self.const_regret(0)

    def test_fp(self):
        self.const_regret(4)

    def const_regret(self, p):
        g = deepcopy(self.game)
        profiles = deepcopy(self.profiles)
        if p != 0:
            attacker = deepcopy(profiles[p])
            attacker.game = g
        else:
            attacker = atk.StochasticAttacker(g, 1, 1)

        defender = bm.FB(g, 0, 1)
        for p in profiles:
            p.game = g
        g.set_players([defender], [attacker], profiles)

        e = runner.Experiment(g)
        e.run()
        last_val = round(e.exp_regret[-1])
        self.assertEqual([round(x) for x in e.exp_regret[-10:]],
                         [last_val for i in range(10)])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
