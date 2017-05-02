import unittest
import logging
import source.player as player
import source.util as util
import source.game as game
import source.players.attackers as atk
import source.runner as runner
from copy import deepcopy


logger = logging.getLogger(__name__)


class BRDefender(player.Defender):
    def __init__(self, g, pl_id, att):
        super().__init__(g, pl_id, 1)
        self.att = att

    def compute_strategy(self, **kwargs):
        return self.br_to(self.att)


class BestResponseTestCase(unittest.TestCase):

    def setUp(self):
        time_horizon = 1000
        targets = util.gen_norm_targets(10)
        self.game = game.zs_game(targets, time_horizon)

    def test_stochastic(self):
        g = deepcopy(self.game)
        attacker = atk.StochasticAttacker(g, 1)
        defender = BRDefender(g, 0, attacker)
        g.set_players([defender], [attacker], [])

        e = runner.Experiment(g)
        e.run()
        self.assertEqual(round(e.exp_regret[-1]), 0)

    def test_stackelberg(self):
        g = deepcopy(self.game)
        attacker = atk.StackelbergAttacker(g, 1)
        defender = BRDefender(g, 0, attacker)
        g.set_players([defender], [attacker], [])

        e = runner.Experiment(g)
        e.run()
        self.assertEqual(round(e.exp_regret[-1]), 0)

    def test_suqr(self):
        g = deepcopy(self.game)
        attacker = atk.SUQR(g, 1)
        defender = BRDefender(g, 0, attacker)
        g.set_players([defender], [attacker], [])

        e = runner.Experiment(g)
        e.run()
        self.assertEqual(round(e.exp_regret[-1]), 0)

    def test_unk_sto(self):
        """
        In this case we have a problem: we cannot guarantee the regret to be
        equal to zero, because the br can give different results if called
        more times in the same round, and we do call it 2 times, when we use
        it and we compute the optimal loss, therefore we can have a little
        regret. We check only that the regret is constant in the final 10
        rounds.
        """
        g = deepcopy(self.game)
        attacker = atk.StochasticAttacker(g, 1)
        defender = BRDefender(g, 0, atk.UnknownStochasticAttacker(g, 1, 1))
        g.set_players([defender], [attacker], [])

        e = runner.Experiment(g)
        e.run()
        last_val = round(e.exp_regret[-1])
        self.assertEqual([round(x) for x in e.exp_regret[-10:]],
                         [last_val for i in range(10)])

    def test_fp(self):
        g = deepcopy(self.game)
        attacker = atk.FictitiousPlayerAttacker(g, 1)
        defender = BRDefender(g, 0, attacker)
        g.set_players([defender], [attacker], [])

        e = runner.Experiment(g)
        e.run()
        self.assertEqual(round(e.exp_regret[-1]), 0)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
