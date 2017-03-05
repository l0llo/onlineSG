import unittest
import source.game as game
import source.player as player


class GameTestCase(unittest.TestCase):

    def setUp(self):
        time_horizon = 10
        self.targets = [1, 2]
        values = tuple((v, v) for v in self.targets)
        self.game = game.Game(values, time_horizon)
        defender = player.Defender(self.game, 0)
        attacker = player.Attacker(self.game, 1)
        other = player.Attacker(self.game, 1)
        self.game.set_players([defender], [attacker], [attacker, other])
        self.game.history = [{0: [0], 1:[0]},
                             {0: [1], 1:[0]},
                             {0: [0], 1:[1]},
                             {0: [1], 1:[1]}]

    def test_payoffs(self):
        payoffs = [{p: self.game.get_player_payoffs(p, h)
                    for p in self.game.players}
                   for h in self.game.history]

        self.assertEqual(payoffs[0][0], [0, 0])
        self.assertEqual(payoffs[1][0], [-1, 0])
        self.assertEqual(payoffs[2][0], [0, -2])
        self.assertEqual(payoffs[3][0], [0, 0])

        self.assertEqual(payoffs[0][1], [0, 0])
        self.assertEqual(payoffs[1][1], [1, 0])
        self.assertEqual(payoffs[2][1], [0, 2])
        self.assertEqual(payoffs[3][1], [0, 0])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
