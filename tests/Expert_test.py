import unittest
import logging
import source.util as util
import source.game as game
import source.players.attackers as atk
from source.players.baseline import Expert

from copy import deepcopy
import tests.def_test as dt

logger = logging.getLogger(__name__)


class ExpertTestCase(dt.DefTestCase):
    """
    verify that the algorithm has a constant regret, namely we check that the
    last 10 values are equal.
    """

    def setUp(self):
        self.conf = True  # enable the configuration testing
        self.n = 10
        time_horizon = 1000
        targets = util.gen_norm_targets(3)
        # choose which adversaries you want to test
        self.adversaries = [dt.Ad.USTO,
                            dt.Ad.STA,
                            dt.Ad.SUQR,
                            dt.Ad.STO,
                            dt.Ad.FP]
        self.game = game.zs_game(targets, time_horizon)
        self.defender = Expert(self.game, 0, 1)
        # choose the profiles
        self.profiles = util.gen_profiles(targets,
                                          [(atk.UnknownStochasticAttacker, 1),
                                           (atk.StackelbergAttacker, 1),
                                           (atk.SUQR, 1),
                                           (atk.StochasticAttacker, 1),
                                           (atk.FictitiousPlayerAttacker, 1)])

    @unittest.skip("")
    def test_regret(self):
        super().test_regret()

#    @unittest.skip("")
    def test_run(self):
        super().test_regret()


if __name__ == '__main__':
    unittest.main()
