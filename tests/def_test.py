import unittest
import logging
import source.util as util
import source.players.attackers as atk
import source.runner as runner
from source.players.baseline import MAB
import concurrent.futures

from copy import deepcopy
#import enum


logger = logging.getLogger(__name__)


#Ad = enum.Enum('Ad', 'USTO STA SUQR STO FP USUQR')
Ad = ["usto", "sta", "suqr", "sto", "fp", "usuqr"]


class DefTestCase(unittest.TestCase):
    """
    base class for Defender subclass testing
    """

    def setUp(self):
        pass

    def test_regret(self):
        for a in self.adversaries:
            with self.subTest(i=a):
                if self.conf:
                    self.regret_conf(a)
                else:
                    self.regret_exp(a)

    def test_run(self):
        for a in self.adversaries:
            with self.subTest(i=a):
                if self.conf:
                    c = self.runconf(a)
                    print(c.stats['exp_regret'])
                else:
                    e = self.runexp(a)
                    print(e.exp_regret)

    def runexp(self, adv):
        g = self.create_game(adv)
        e = runner.Experiment(g)
        e.run()
        return e

    def runconf(self, adv):
        g = self.create_game(adv)
        c = runner.Configuration(g, print_results=False)
        with concurrent.futures.ProcessPoolExecutor(None) as executor:
            futures = {}
            c.run(futures, executor, n=self.n)
        c.collect(futures)
        return c

    def create_game(self, adv):
        g = deepcopy(self.game)
        profiles = deepcopy(self.profiles)
        # for p in profiles:
        #     p.game = g
        # if (adv.value - 1) == 0:  # enum start from 1
        #     attacker = atk.StochasticAttacker(g, 1, 1)
        # elif (adv.value - 1) == 0:
        #     attacker = atk.USUQR(g, 1)
        # else:
        #     attacker = deepcopy(profiles[adv.value - 1])
        #     attacker.game = g
        for p in profiles:
            p.game = g
        if self.pdict[adv].adv is not None:
            attacker = deepcopy(self.pdict[adv].adv)
        else:
            attacker = deepcopy(self.pdict[adv].prof)
        attacker.game = g
        defender = deepcopy(self.defender)
        defender.game = g
        g.set_players([defender], [attacker], profiles)
        return g

    def regret_exp(self, adv):
        e = self.runexp(adv)
        name = self.defender.__class__.name + " vs " + str(adv)
        d = {"name": name, "avgs": e.exp_regret}
        util.plot_dicts([d], save=False, show=True)

    def regret_conf(self, adv):
        c = self.runconf(adv)
        name = self.defender.__class__.name + " vs " + str(adv)
        d = {"name": name, "avgs": c.stats["exp_regret"],
             "lb": c.stats["lb_exp_regret"], "ub": c.stats["ub_exp_regret"]}
        util.plot_dicts([d], save=False, show=True, semilog=False)

    def tearDown(self):
        pass


# if __name__ == '__main__':
#     unittest.main()
