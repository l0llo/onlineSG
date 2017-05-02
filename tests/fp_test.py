import unittest
import logging
import source.util as util
import source.game as game
import source.players.attackers as atk
import source.runner as runner
import source.player as player
from source.players.fr import FR
from copy import deepcopy


logger = logging.getLogger(__name__)


class FP_opt_lossTestCase(unittest.TestCase):
    """
    verify that the algorithm has a constant regret, namely we check that the
    last 10 values are equal.
    """

    def setUp(self):
        time_horizon = 2
        targets = [1, 2, 3]
        self.game = game.zs_game(targets, time_horizon)
        attacker = atk.FictitiousPlayerAttacker(self.game, 1, 1)
        defender = player.Defender(self.game, 0, 1)
        self.game.set_players([defender], [attacker], [])

    def test_compute_strategy(self):
        H = [(0, 0), (1, 0), (2, 0)]
        ol = self.game.players[1].opt_loss(history=H)
        self.assertEqual(ol, 1/4)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
