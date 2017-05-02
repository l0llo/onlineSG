import unittest
import logging
import source.util as util
import source.game as game
import source.players.attackers as atk
import source.runner as runner
import source.players.belief_max as bm
from copy import deepcopy


logger = logging.getLogger(__name__)


class MemorySUQRTestCase(unittest.TestCase):
    """
    verify that the use of memory is useful to save time.
    """

    def setUp(self):
        time_horizon = 1000
        targets = util.gen_norm_targets(10)
        self.game = game.zs_game(targets, time_horizon)
        self.profiles1 = util.gen_profiles(targets,
                                           [(atk.UnknownStochasticAttacker, 1),
                                            (atk.StackelbergAttacker, 1),
                                            (atk.StochasticAttacker, 5),
                                            (atk.FictitiousPlayerAttacker, 1)])
        self.profiles2 = util.gen_profiles(targets, [(atk.SUQR, 5)])
        self.profiles3 = util.gen_profiles(targets, [(atk.SUQR, 5)])
        for p in self.profiles3:
            p._use_memory = False

    def test_time(self):

        # WITH MEMORY
        g = deepcopy(self.game)
        profiles = deepcopy(self.profiles1) + self.profiles2
        # attacker is one of the SUQR profiles
        attacker = deepcopy(self.profiles2[-1])
        attacker.game = g
        defender = bm.FB(g, 0, 1)
        for p in profiles:
            p.game = g
        g.set_players([defender], [attacker], profiles)
        c1 = runner.Configuration(g, print_results=False)
        c1.run(n=10)

        # WITH MEMORY
        g = deepcopy(self.game)
        profiles = deepcopy(self.profiles1) + self.profiles3
        # attacker is one of the SUQR profiles
        attacker = deepcopy(self.profiles3[-1])
        attacker.game = g
        defender = bm.FB(g, 0, 1)
        for p in profiles:
            p.game = g
        g.set_players([defender], [attacker], profiles)
        c2 = runner.Configuration(g, print_results=False)
        c2.run(n=10)

        self.assertGreater(c2.stats["avg_run_time"], c1.stats["avg_run_time"])
        print(c2.stats["avg_run_time"], c1.stats["avg_run_time"])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
