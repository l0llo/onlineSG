import unittest
import logging
import source.game as game
import source.players.attackers as atk
import source.util as util
from copy import deepcopy
import source.parsers as parsers
import source.players.belief_max as bm
import source.players.fr as fr
import source.players.baseline as bl


logger = logging.getLogger(__name__)


players = {"sto1-0.1-0.7-0.2": atk.StochasticAttacker,
           "sto1": atk.StochasticAttacker,
           "sta": atk.StackelbergAttacker,
           "usto": atk.UnknownStochasticAttacker,
           "suqr": atk.SUQR,
           "usuqr": atk.USUQR,
           "suqr1-15-0.8": atk.SUQR,
           "fictitious": atk.FictitiousPlayerAttacker,
           "FB1": bm.FB,
           "FR1-1": fr.FR,
           "MAB1": bl.MAB,
           "EXP1": bl.Expert}

att_classes = [atk.StochasticAttacker,
               atk.StackelbergAttacker,
               atk.UnknownStochasticAttacker,
               atk.SUQR,
               atk.FictitiousPlayerAttacker,
               atk.USUQR]


class ParsingTestCase(unittest.TestCase):
    """

    """

    def setUp(self):
        time_horizon = 1
        targets = [1, 2, 3]
        self.game = game.zs_game(targets, time_horizon)

    def test_players(self):
        for p in players:
            with self.subTest(i=p):
                self.parse_player(p, players[p])

    def test_player_str(self):
        targets = [1, 2, 3]
        for a in att_classes:
            with self.subTest(i=a):
                att = util.gen_profiles(targets, [(a, 1)])[0]
                self.parse_player(str(att), a)

    def parse_player(self, string, cls):
        pl = parsers.parse_player(string, self.game, 1)
        self.assertIsInstance(pl, cls)


if __name__ == '__main__':
    unittest.main()
