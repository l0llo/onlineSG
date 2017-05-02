import unittest
import source.util as util
from gurobipy import *
import scipy.optimize
import numpy as np
import logging


logger = logging.getLogger(__name__)


class LinProgTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_linprog(self):
        L = 10
        J = 10
        for l in range(2, L):
            for i in range(J):
                values = util.gen_norm_targets(l)
                g_sol = gurobi_solve(values)
                logger.debug("gurobi: " + str(g_sol))
                s_sol = scipy_solve(values)
                logger.debug("scipy: " + str(s_sol))
                self.assertEqual(g_sol, s_sol)

    def tearDown(self):
        pass


def scipy_solve(values):
    A_ub = []
    targets = list(range(len(values)))
    for t in targets:
        terms = [values[t] * int(i != t) for i in targets]
        terms += [1]
        A_ub.append(terms)
    b_ub = [0 for i in range(len(A_ub))]
    A_eq = [[1 for i in targets] + [0]]
    b_eq = [1]
    bounds = [(0, 1) for i in range(len(values))] + [(None, None)]
    scipy_sol = list(scipy.optimize.linprog([0 for i in targets] + [-1],
                                            A_ub=np.array(A_ub),
                                            b_ub=np.array(b_ub),
                                            A_eq=np.array(A_eq),
                                            b_eq=np.array(b_eq),
                                            bounds=bounds,
                                            method='simplex').x)
    return [round(j, 4) for j in scipy_sol]


def gurobi_solve(values):
    m = Model("SSG")
    targets = list(range(len(values)))
    strategy = []
    for t in targets:
        strategy.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + str(t)))
    v = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")
    m.setObjective(v, GRB.MAXIMIZE)
    for t in targets:
        terms = [-values[t] * strategy[i] for i in targets if i != t]
        m.addConstr(sum(terms) - v >= 0, "c" + str(t))
    m.addConstr(sum(strategy) == 1, "c" + str(len(targets)))
    m.params.outputflag = 0
    m.optimize()
    gurobi_sol = [float(s.x) for s in strategy] + [m.objVal]
    return [round(j, 4) for j in gurobi_sol]


if __name__ == '__main__':
    unittest.main()
