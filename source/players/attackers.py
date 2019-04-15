import source.player as player
import source.standard_player_parsers as spp
import re
import source.errors as errors
import numpy as np
from math import exp, sqrt, log
import source.util as util
import scipy.optimize
import logging
import source.game as game

logger = logging.getLogger(__name__)


class StrategyAwareAttacker(player.Attacker):

    def exp_loss(self, strategy_vec, **kwargs):
        """
        exp loss for a strategy aware attacker
        """
        if isinstance(strategy_vec, dict):
            # logger.debug(str(strategy_vec) + " " + str(type(strategy_vec)))
            # only for testing purposes
            return super().exp_loss(strategy_vec, **kwargs)
        else:
            str_dict = {0: strategy_vec,
                        1: self.compute_strategy(strategy=strategy_vec)}
            return super().exp_loss(str_dict)

    def opt_loss(self, **kwargs):
        """
        kwargs in this case are useful only for learner/unknown parameters
        attackers

        """
        if not kwargs and self.last_ol is not None:
            return self.last_ol
        else:
            d_strat = self.best_response(**kwargs)
            kwargs['strategy'] = d_strat
            a_strat = self.compute_strategy(**kwargs)
            s = {0: d_strat,
                 1: a_strat}
            self.last_ol = self.exp_loss(s)
            return self.last_ol


class HistoryDependentAttacker(player.Attacker):

    def opt_loss(self, history=None, **kwargs):
        """
        being history dependent, it should be recomputed every time
        """
        if not history:
            d_strat = self.best_response(**kwargs)
            a_strat = self.compute_strategy(**kwargs)
            s = {0: d_strat,
                 1: a_strat}
            self.last_ol = self.exp_loss(s)
            return self.last_ol
        else:
            d_strat = self.best_response(history=history, **kwargs)
            a_strat = self.compute_strategy(history=history, **kwargs)
            s = {0: d_strat,
                 1: a_strat}
            ol = self.exp_loss(s)
            h1 = history[:-1] if history[:-1] else None
            weight = len(h1) + 1 if h1 else 1
            ol1 = self.opt_loss(history=h1) * weight
            self.last_ol = ol + ol1 / (len(history) + 1)
            return self.last_ol


class StackelbergAttacker(StrategyAwareAttacker):
    """
    The Stackelberg attacker observes the Defender strategy and plays a pure
    strategy that best responds to it.
    """

    name = "sta"
    pattern = re.compile(r"^" + name + "\d*$")

    def compute_strategy(self, strategy=None, **kwargs):
        if strategy is None:
            return self.best_respond(self.game.strategy_history[-1])
        else:
            return self.best_respond(strategy)

    def best_response(self, **kwargs):
        if not self.last_br:
            # m = gurobipy.Model("SSG")
            # targets = list(range(len(self.game.values)))
            # strategy = []
            # for t in targets:
            #     strategy.append(m.addVar(vtype=gurobipy.GRB.CONTINUOUS,
            #                              name="x" + str(t)))
            # v = m.addVar(lb=-gurobipy.GRB.INFINITY,
            #              vtype=gurobipy.GRB.CONTINUOUS, name="v")
            # m.setObjective(v, gurobipy.GRB.MAXIMIZE)
            # for t in targets:
            #     terms = [-self.game.values[t][0] * strategy[i]
            #              for i in targets if i != t]
            #     m.addConstr(sum(terms) - v >= 0, "c" + str(t))
            # m.addConstr(sum(strategy) == 1, "c" + str(len(targets)))
            # m.params.outputflag = 0
            # m.optimize()
            # self.last_ol = -v.x
            # self.last_br = [float(s.x) for s in strategy]

            A_ub = []
            for t in self.M:
                terms = [self.game.values[t][self.id] * int(i != t)
                         for i in self.M]
                terms += [1]
                A_ub.append(terms)
            b_ub = [0 for i in range(len(A_ub))]
            A_eq = [[1 for i in self.M] + [0]]
            b_eq = [1]
            bounds = [(0, 1) for i in self.M] + [(None, None)]
            scipy_sol = list(scipy.optimize.linprog([0 for i in self.M] + [-1],
                                                    A_ub=np.array(A_ub),
                                                    b_ub=np.array(b_ub),
                                                    A_eq=np.array(A_eq),
                                                    b_eq=np.array(b_eq),
                                                    bounds=bounds,
                                                    method='simplex').x)

            self.last_br, self.last_ol = scipy_sol[:-1], -scipy_sol[-1]
        return self.last_br

    def best_response_with_obs(self, **kwargs):
        if not self.last_br:
            A_ub = []
            for t in self.M:
                terms = [self.game.values[t][self.id] * (int(i != t) if i != t or not isinstance(self.game, game.GameWithObservabilities)
                                                                   else 1 - self.game.observabilities.get(t))
                         for i in self.M]
                terms += [1]
                A_ub.append(terms)
            b_ub = [0 for i in range(len(A_ub))]
            A_eq = [[1 for i in self.M] + [0]]
            b_eq = [1]
            bounds = [(0, 1) for i in self.M] + [(None, None)]
            scipy_sol = list(scipy.optimize.linprog([0 for i in self.M] + [-1],
                                                    A_ub=np.array(A_ub),
                                                    b_ub=np.array(b_ub),
                                                    A_eq=np.array(A_eq),
                                                    b_eq=np.array(b_eq),
                                                    bounds=bounds,
                                                    method='simplex').x)

            self.last_br, self.last_ol = scipy_sol[:-1], -scipy_sol[-1]
        return self.last_br

    # def init_br(self):
    #     br = base_defenders.StackelbergDefender(self.game, 0)
    #     return br


class FictitiousPlayerAttacker(HistoryDependentAttacker):
    """
    The fictitious player computes the empirical distribution of the
    adversary move and then best respond to it. When it starts it has a vector
    of weights for each target and at each round the plays the inverse of that
    weight normalized to the weights sum. Then he observe the opponent's move
    and update the weights acconding to it.
    """
    name = "fictitious"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1, initial_weight=0):
        super().__init__(game, id, resources)
        self.initial_weight = initial_weight
        self.weights = [initial_weight for m in self.M]

    def compute_strategy(self, history=None, **kwargs):
        """
        Add 1 to the weight of each covered target in the defender profile
        at each round: then best respond to the computed strategy
        """

        if history:
            add_w = [0 for m in self.M]
            for i, j in history:
                add_w[i] += 1
            weights = [add_w[m] + self.weights[m] for m in self.M]
            norm = sum(weights)
            weights = [w / norm for w in weights]
            wdict = {0: weights}
        else:
            norm = sum(self.weights)
            if norm == 0:
                return self.uniform_strategy(len(self.game.values))
            weights = [w / norm for w in self.weights]
            wdict = {0: self.weights}
        return self.best_respond(wdict)

    def learn(self):
        t = self.game.history[-1][0][0]
        self.weights[t] += 1


class StochasticAttacker(player.Attacker):
    """
    It attacks according to a fixed
    """

    name = "sto"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            arguments = [float(a) for a in
                         player_type.split(cls.name)[1].split("-")
                         if a != '']
            if not arguments:
                return cls(game, id)
            elif len(arguments) == 1:
                return cls(game, id, int(arguments[0]))
            else:
                arguments[0] = int(arguments[0])
                if (len(arguments) == len(game.values) + 1):
                    is_prob = round(sum(arguments[1:]), 3) == 1
                    if is_prob:
                        args = [game, id] + arguments
                        return cls(*args)
                    else:
                        raise errors.NotAProbabilityError(arguments[1:])

    def __init__(self, g, id, resources=1, *distribution):
        super().__init__(g, id, resources)
        if not distribution:
            self.distribution = util.gen_distr(len(g.values))
        else:
            self.distribution = list(distribution)

    def compute_strategy(self, **kwargs):
        return self.distribution

    # def init_br(self):
    #     br = base_defenders.KnownStochasticDefender(self.game, 0, 1,
    #                                                 *self.distribution)
    #     return br

    def __str__(self):
        return "-".join([super().__str__()] +
                        [str(d) for d in self.distribution])


class UnknownStochasticAttacker(HistoryDependentAttacker):
    """
    Not a real attacker to be instantiated: it is intended to be used by the
    defender as a model
    """

    name = "usto"
    pattern = re.compile(r"^" + name + "\d*(-\d+(\.\d+)?)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, game, id, resources=1, lb=0):
        super().__init__(game, id, resources)
        self.weights = [0 for m in self.M]
        self.lb = lb

    def compute_strategy(self, hdict=None, **kwargs):
        """
        Add 1 to the weight of each covered target in the defender profile
        at each round: then best respond to the computed strategy
        """

        if self.tau() == 0:
            return self.uniform_strategy(len(self.game.values))
        else:
            if hdict is not None:
                distr = util.norm_min(hdict["weights"], m=self.lb)
            else:
                # norm = sum(self.weights)
                distr = util.norm_min(self.weights, m=self.lb)
            return distr

    def learn(self):
        t = self.game.history[-1][self.id][0]
        self.weights[t] += 1

    def hlearn(self, H, ds_history, hdict):
        add_w = [0 for m in self.M]
        for i, j in H:
            add_w[j] += 1
        weights = [add_w[m] + self.weights[m] for m in self.M]
        return {"weights": weights}

    def best_response(self, **kwargs):
        """
        best reponse to an unknown stochastic attacker using FPL. If
        `rep` keyword is not set, it returns a pure strategy,otherwise it
        returns a mixed strategy obtained by averaging over `rep` samples of
        computed strategies.
        Due to the  randomization involved in the br computation, calling
        more times this function in the same round could give different
        results.
        """

        N = len(self.M)
        norm_const = max([v[self.id] for v in self.game.values])
        # if I am seeing a br in the "future" I have to compute the correct t
        add_t = len(kwargs["history"]) if "history" in kwargs else 0
        adj_t = self.tau() + add_t + 1

        def noise():
            return np.random.uniform(0, (norm_const * sqrt(N) / adj_t))
        weights = [0 for i in self.M]
        repetitions = kwargs["rep"] if "rep" in kwargs else 1
        for i in range(repetitions):
            m = min(self.M,
                    key=lambda t:
                    self.exp_loss(self.ps(t), **kwargs) + noise())
            weights[m] += 1
        norm = sum(weights)
        weights = [w / norm for w in weights]
        self.last_br = weights
        return self.last_br

    def opt_loss(self, **kwargs):
        return player.Attacker.opt_loss(self, **kwargs)

    def loglk(self, old_loglk):
        ll = sum([self.weights[m] * log(self.last_strategy[m])
                  for m in self.M if self.last_strategy[m]])
        return ll / self.tau()

    def hloglk(self, old_loglk, hdict,
               history, ds_history):
        t = len(self.game.strategy_history) + len(ds_history)
        ll = sum([hdict["weights"][m] * log(hdict["last_strategy"][m])
                  for m in self.M if self.last_strategy[m]])
        return ll / t

    def get_attacker(self):
        return StochasticAttacker(self.game, 1)

class BayesianUnknownStochasticAttacker(player.Attacker):

    name = "busto"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        if cls.pattern.match(player_type):
            arguments = [float(a) for a in
                         player_type.split(cls.name)[1].split("-")
                         if a != '']
            if not arguments:
                return cls(game, id)
            elif len(arguments) == 1:
                return cls(game, id, int(arguments[0]))
            else:
                arguments[0] = int(arguments[0])
                if (len(arguments) == len(game.values) + 1):
                    is_prob = round(sum(arguments[1:]), 3) == 1
                    if is_prob:
                        args = [game, id] + arguments
                        return cls(*args)
                    else:
                        raise errors.NotAProbabilityError(arguments[1:])

    def __init__(self, g, id, resources=1, *distribution):
        super().__init__(g, id, resources)
        if not distribution:
            self.actual_distribution = util.gen_distr(len(g.values))
        else:
            self.actual_distribution = list(distribution)
        num_t = len(self.game.values)
        self.empirical_distribution = [1/num_t for t in range(num_t)]

    def compute_strategy(self, **kwargs):
        return self.actual_distribution

    def compute_empirical_strategy(self, **kwargs):
        return self.empirical_distribution

    def set_weights(self, weights):
        if sum(weights) != 1:
            norm = sum(weights)
            self.empirical_distribution = [w/norm for w in weights]
        self.empirical_distribution = weights

    def __str__(self):
        return "-".join([super().__str__()] +
                        [str(d) for d in self.distribution])

    def exp_loss(self, strategy_vec, **kwargs):
        if isinstance(strategy_vec, dict):
            return super().exp_loss(strategy_vec, **kwargs)
        else:
            return super().exp_loss({0: strategy_vec, 1: self.compute_empirical_strategy(**kwargs)})

    def opt_loss(self, **kwargs):
        if not kwargs and self.last_ol is not None:
            return self.last_ol
        else:
            d_strat = self.actual_best_response(**kwargs)
            a_strat = self.compute_strategy(**kwargs)
            s = {0: d_strat,
                 1: a_strat}
            self.last_ol = self.exp_loss(s)
            return self.last_ol

    def actual_best_response(self, **kwargs):
        m = min(self.M, key=lambda t: self.exp_loss({0: self.ps(t), 1: self.compute_strategy()}, **kwargs))
        return self.ps(m)

#    def best_response(self, **kwargs):
#        m = min(self.M, key=lambda t: self.exp_loss(self.ps(t), **kwargs))
#        return self.ps(m)

class SUQR(StrategyAwareAttacker):

    name = "suqr"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?){2})?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, g, pl_id, use_memory=True, w1=None,
                 w2=None):
        super().__init__(g, pl_id, 1)

        if w1 is None:
            self.w1 = round(np.random.uniform(5, 15), 3)
        else:
            self.w1 = w1
        if w2 is None:
            self.w2 = round(np.random.uniform(0, 1), 3)
        else:
            self.w2 = w2
        self._use_memory = use_memory
        self.memory = dict()

    def compute_strategy(self, strategy=None, **kwargs):
        if strategy is None:
            x = self.game.strategy_history[-1][0]
            return self.qr(x)
        else:
            return self.qr(strategy)

    def qr(self, x, a=None, b=None):
        if self._use_memory:
            if tuple(x) in self.memory:
                return self.memory[tuple(x)]
        targets = list(range(len(self.game.values)))
        R = [v[self.id] for v in self.game.values]
        if a is not None and b is not None:
            q = np.array([exp((-a * x[t] +
                               b * R[t]))
                          for t in targets])
        else:
            q = np.array([exp((-self.w1 * x[t] +
                               self.w2 * R[t]))
                          for t in targets])
        q /= np.linalg.norm(q, ord=1)
        if self._use_memory:
            self.memory[tuple(x)] = list(q)
        return list(q)

    def best_response(self, **kwargs):
        if self.last_br is None:
            def fun(x):
                return self.exp_loss(x)
            targets = list(range((len(self.game.values))))
            bnds = tuple([(0, 1) for t in targets])
            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
            res = scipy.optimize.minimize(fun, util.gen_distr(len(targets)),
                                          method='SLSQP', bounds=bnds,
                                          constraints=cons, tol=0.000001)
            self.last_br = list(res.x)
        return self.last_br

    def __str__(self):
        return "-".join([super().__str__()] +
                        [str(self.w1), str(self.w2)])


class USUQR(SUQR):

    name = "usuqr"
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, g, pl_id, mle=True):
        super().__init__(g, pl_id, 1)
        self._use_memory = False
        self.mle = bool(mle)
        self.last_w1 = None
        self.last_w2 = None

    def compute_strategy(self, strategy=None, hdict=None, **kwargs):
        if hdict is not None:
            return self.qr(strategy, a=hdict["w"][0], b=hdict["w"][1])
        else:
            return super().compute_strategy(strategy=strategy)

    def best_response(self, **kwargs):
        self.last_br = None
        return super().best_response(**kwargs)

    def stochastic_gradient_descent(self, history=None,
                                    ds_history=None, hdict=None):
        if hdict is not None:
            if "w" in hdict:
                old_w1, old_w2 = hdict["w"]
            else:
                old_w1, old_w2 = self.w1, self.w2
            s = ds_history[-1]
            j = history[-1][1]
        else:
            old_w1, old_w2 = self.w1, self.w2
            s = self.game.strategy_history[-1][0]
            j = self.game.history[-1][1][0]
        v = [x[0] for x in self.game.values]
        factors = [exp(-old_w1 * s[i] + old_w2 * v[i])
                   for i in range(len(v))]
        den = sum(factors)
        num1 = sum([(s[i] - s[j]) * f for i, f in enumerate(factors)])
        num2 = sum([(v[j] - v[i]) * f for i, f in enumerate(factors)])
        gr1, gr2 = num1 / den, num2 / den
        eta = 0.5
        w1 = min(max(old_w1 + eta * gr1, 5), 15)
        w2 = min(max(old_w2+ eta * gr2, 0), 1)
        return w1, w2

    def weights_MLE(self, history=None, ds_history=None):

        bnds = tuple([(5, 15), (0, 1)])
        na, nb = 0, 0

        def fun(w):
            return self.neg_loglk(w, history, ds_history)
        num_ite = 1
        for ite in range(num_ite):
            x0 = [np.random.uniform(5, 15), np.random.uniform(0, 1)]
            res = scipy.optimize.minimize(fun=fun,
                                          x0=x0,
                                          method='L-BFGS-B',
                                          bounds=bnds)
            na = na + (res.x[0] - na) / (ite + 1)
            nb = nb + (res.x[1] - nb) / (ite + 1)
        return na, nb

    def learn(self):
        if self.mle:
            self.w1, self.w2 = self.weights_MLE()
        else:
            self.w1, self.w2 = self.stochastic_gradient_descent()
        #self.last_br = None

    def hlearn(self, history, ds_history, hdict):
        hdict = {}
        if self.mle:
            hdict["w"] = self.weights_MLE(history, ds_history)
        else:
            hdict["w"] = self.stochastic_gradient_descent(history, ds_history, hdict)
        return hdict
        #self.last_br = None

    def loglk(self, old_loglk):
        ll = - self.neg_loglk([self.w1, self.w2])
        return ll / self.tau()

    def hloglk(self, old_loglk, hdict,
               history, ds_history):
        t = len(self.game.strategy_history) + len(ds_history)
        ll = - self.neg_loglk(hdict["w"], history, ds_history)
        return ll / t

    def neg_loglk(self, w, history=None, ds_history=None):
        ll = 0
        for i, strat in enumerate(self.game.strategy_history):
            s = strat[0]
            j = self.game.history[i][1][0]
            ll -= log(self.qr(s, w[0], w[1])[j])
        if history is not None and ds_history is not None:
            for i, s in enumerate(ds_history):
                j = history[i][1]
                ll -= log(self.qr(s, w[0], w[1])[j])
        return ll

    def get_attacker(self):
        return SUQR(self.game, 1)

    def __str__(self):
        if not self.mle:
            return self.__class__.name + "0"
        else:
            return self.__class__.name

class ObservingStrategyAwareAttacker(player.ObservingAttacker):

    def exp_loss(self, strategy_vec, **kwargs):
        """
        exp loss for a strategy aware attacker
        """
        if isinstance(strategy_vec, dict):
            # logger.debug(str(strategy_vec) + " " + str(type(strategy_vec)))
            # only for testing purposes
            return super().exp_loss(strategy_vec, **kwargs)
        else:
            str_dict = {0: strategy_vec,
                        1: self.compute_strategy(strategy=strategy_vec)}
            return super().exp_loss(str_dict)

    def opt_loss(self, **kwargs):
        """
        kwargs in this case are useful only for learner/unknown parameters
        attackers

        """
        if not kwargs and self.last_ol is not None:
            return self.last_ol
        else:
            d_strat = self.best_response(**kwargs)
            kwargs['strategy'] = d_strat
            a_strat = self.compute_strategy(**kwargs)
            s = {0: d_strat,
                 1: a_strat}
            self.last_ol = self.exp_loss(s)
            return self.last_ol

class ObservingStackelbergAttacker(ObservingStrategyAwareAttacker):
    """
    This attacker takes into account the defender strategy as the Stackelberg does,
    but also the observation probabilities of the targets.
    """

    name = "obsta"
    pattern = re.compile(r"^" + name + "\d*$")

    def compute_strategy(self, strategy=None, **kwargs):
        if strategy is None:
            return self.best_respond(self.game.strategy_history[-1])
        else:
            return self.best_respond(strategy)

    def best_response(self, **kwargs):
        if not self.last_br:
            A_ub = []
            for t in self.M:
                terms = [self.game.values[t][self.id] * int(i != t if i != t or not isinstance(self.game, game.GameWithObservabilities)
                                                                   else 1 - self.game.observabilities.get(t))
                         for i in self.M]
                terms += [1]
                A_ub.append(terms)
            b_ub = [0 for i in range(len(A_ub))]
            A_eq = [[1 for i in self.M] + [0]]
            b_eq = [1]
            bounds = [(0, 1) for i in self.M] + [(None, None)]
            scipy_sol = list(scipy.optimize.linprog([0 for i in self.M] + [-1],
                                                    A_ub=np.array(A_ub),
                                                    b_ub=np.array(b_ub),
                                                    A_eq=np.array(A_eq),
                                                    b_eq=np.array(b_eq),
                                                    bounds=bounds,
                                                    method='simplex').x)

            self.last_br, self.last_ol = scipy_sol[:-1], -scipy_sol[-1]
        return self.last_br

class ObservingSUQR(ObservingStrategyAwareAttacker):

    name = "obsuqr"
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?){2})?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, g, pl_id, use_memory=True, w1=None,
                 w2=None, w3=None):
                 super().__init__(g, pl_id, use_memory, w1, w2)
                 if w3 is None:
                     self.w3 = round(np.random.uniform(0, 1), 3) #TODO: extreme values need to be checked
                 else:
                     self.w3 = w3

    def compute_strategy(self, strategy=None, **kwargs):
        if strategy is None:
            x = self.game.strategy_history[-1][0]
            return self.qr(x)
        else:
            return self.qr(strategy)

    def qr(self, x, a=None, b=None, c=None):
        if self._use_memory:
            if tuple(x) in self.memory:
                return self.memory[tuple(x)]
        targets = list(range(len(self.game.values)))
        R = [v[self.id] for v in self.game.values]
        if isinstance(self.game, game.GameWithObservabilities):
            if a is not None and b is not None and c is not None:
                q = np.array([exp((-a * x[t] +
                                b * R[t]
                                -c * self.game.observabilities.get(t)))
                                for t in targets])
            else:
                q = np.array([exp((-self.w1 * x[t] +
                                self.w2 * R[t]
                                -self.w3 * self.game.observabilities.get(t)))
                                for t in targets])
        else:
            if a is not None and b is not None and c is not None:
                q = np.array([exp((-a * x[t] +
                                b * R[t]))
                                for t in targets])
            else:
                q = np.array([exp((-self.w1 * x[t] +
                                self.w2 * R[t]))
                                for t in targets])

        q /= np.linalg.norm(q, ord=1)
        if self._use_memory:
            self.memory[tuple(x)] = list(q)
        return list(q)

        def best_response(self, **kwargs):
            if self.last_br is None:
                def fun(x):
                    return self.exp_loss(x)
                targets = list(range((len(self.game.values))))
                bnds = tuple([(0, 1) for t in targets])
                cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
                res = scipy.optimize.minimize(fun, util.gen_distr(len(targets)),
                                              method='SLSQP', bounds=bnds,
                                              constraints=cons, tol=0.000001)
                self.last_br = list(res.x)
            return self.last_br

        def __str__(self):
            return "-".join([super().__str__()] +
                            [str(self.w1), str(self.w2), str(self.w3)])
