import unittest
import logging
import source.util as util
import source.game as game
import source.players.attackers as atk
import source.player as player
from source.players.fr import FR
from copy import deepcopy
import enum


logger = logging.getLogger(__name__)


Ad = enum.Enum('Ad', 'USTO STA SUQR STO FP')


class FRTestCase(unittest.TestCase):
    """
    verify that the algorithm has a constant regret, namely we check that the
    last 10 values are equal.
    """

    H_MAX = 1

    def setUp(self):
        time_horizon = 100
        targets = [1, 2, 3]
        self.game = game.zs_game(targets, time_horizon)
        attacker = atk.StackelbergAttacker(self.game, 1, 1)
        defender = player.Defender(self.game, 0, 1)
        self.game.set_players([defender], [attacker], [])

    def test_regret_usto(self):
        self.assertEqual(len(util.support(self.game)), 2)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
