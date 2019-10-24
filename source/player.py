import numpy as np
import re
from copy import deepcopy
import enum
from source.errors import AlreadyFinalizedError
from math import log
import source.game as game
import scipy
#: The possible type of learning of a learner player
# Learning = enum.Enum('Learning', 'MAB EXPERT OTHER')


class Player:
    """
    Base class from which all players inherit. It implements some methods
    that every player shares as :meth:`sample_strategy`, :meth:`finalize_init`
    , :meth:`tau` and :meth:`receive_feedback`

    Each subclass has class attributes (name and pattern) and a class
    method (parse): they are used for the parsing of the player columns in
    the config files. Player itself should not be ever instanciated

    :var game: the Game object to which the player participates (could be
               a real player or simply a profile, or a model)
    :var id: the id that identifies the player in the Game object: if the
             player is a profile then, it has the id of the player who is
             trying to model
    :var resources: the number of targets that can be involved in a move
                    by this player
    :var last_strategy: last played (not only computed) strategy by the
                         player
    :var _finalized: whether :meth:`finalize_init` has been called or not
    :var M: the list of targets indexes
    """


    #: the string the player will be idenfified with in
    #: configuration files
    name = "player"
    #: the pattern the parse method will use to recognize a player
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        """
        method called by the parser when parsing the players
        columns in the configuration file
        """
        if cls.pattern.match(player_type):
            args = [game, id] + [int(a) for a in
                                 player_type.split(cls.name)[1].split("-")
                                 if a != '']
            return cls(*args)
        else:
            return None

    def __init__(self, game, pl_id, resources=1):
        self.game = game
        self.id = pl_id
        self.resources = resources
        self.last_strategy = None
        self._finalized = False
        self.M = list(range(len(self.game.values)))
        self.A = None
        self.V = None
        self.last_el = None

    def finalize_init(self):
        # define some aliases
        self.V = [v[self.id] for v in self.game.values]
        self.A = self.game.profiles
        if not self._finalized:
            self._finalized = True
        else:
            raise AlreadyFinalizedError(self)

    def tau(self):
        """
        returns how many rounds have been passed since the beginning of the
        game
        """
        return len(self.game.history)

    def play_strategy(self, **kwargs):
        """
        calls :meth:`compute_strategy` and then saves the obtained strategy in
        *last_strategy*
        """
        self.last_strategy = self.compute_strategy(**kwargs)
        return self.last_strategy

    def compute_strategy(self, **kwargs):
        """
        sets a probability distribution over the targets
        default: uniform strategy
        """
        return self.uniform_strategy(len(self.game.values))

    def sample_strategy(self):
        """
        samples a move from the computed distribution
        """
        return sample(self.last_strategy, self.resources)

    def uniform_strategy(self, elements):
        return [self.resources / elements
                for i in range(elements)]

    def learn(self):
        pass

    def receive_feedback(self, feedback=None):
        """
        manages the feedbacks obtained by the Environment, calling also
        the learn method
        """
        self.learn()

        # if hasattr(self, "learning"):
        #     self.learn()
        #     if self.learning == Learning.MAB:
        #         if self.sel_arm is not None:
        #             self.sel_arm.receive_feedback()
        #     elif self.learning == Learning.EXPERT:
        #         if (isinstance(self.arms, list) or
        #                 isinstance(self.arms, tuple)):
        #             for a in self.arms:
        #                 a.receive_feedback()
        #         elif isinstance(self.arms, dict):
        #             for a in self.arms:
        #                 self.arms[a].receive_feedback()


    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " id:", str(self.id),
                        " resources:", str(self.resources), ">"])

    def __str__(self):
        return self.__class__.name + str(self.resources)

    def _json(self):
        self_copy = deepcopy(self)
        d = self_copy.__dict__
        d.pop('game', None)
        d["class_name"] = self.__class__.__name__
        return d

    def ps(self, target):
        """
        return the pure strategy on the passed target
        """
        return tuple([int(i == target) for i in self.M])


class Defender(Player):
    """
    """

    name = "defender"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1):
        """"
        Attributes

        feedbacks   list of targets dict with feedbacks for each turn
                    (if any)
        """
        super().__init__(game, id, resources)
        self.feedbacks = []

    def receive_feedback(self, feedback):
        if feedback is not None:
            self.feedbacks.append(feedback)
        super().receive_feedback(feedback)

    def last_reward(self):
        return sum(self.feedbacks[-1].values())

    def br_uniform(self):
        targets = range(len(self.game.values))
        max_target = max(targets, key=lambda x: self.game.values[x][self.id])
        return [int(i == max_target) for i in targets]

    def br_to(self, a, **kwargs):
        if (isinstance(self.game, game.PartialFeedbackGame) and
                        any([int(o) != 1 for o in
                            self.game.observabilities.values()])):
            try:
                a.best_response_with_obs(**kwargs)
            except:
                a.best_response(**kwargs)
        return a.best_response(**kwargs)

    def multi_lp_br_to(self, profiles, **kwargs):
        A_eq = [[1 for i in self.M] + [0]]
        b_eq = [self.resources]
        A_ub = []
        for t in self.M:
            terms = [self.game.values[t][profiles[0].id] * int(i != t)
                     for i in self.M]
            terms += [1]
            A_ub.append(terms)
        b_ub = [0 for i in range(len(A_ub))]
        bounds = [(0, 1) for i in self.M] + [(None, None)]
        c = [0 for i in self.M] + [0]
        for p in profiles:
            consts = p.update_obj_fun(1)
            c = [c[i] + consts[i] for i in range(len(c))]
        scipy_sol = list(scipy.optimize.linprog(c,
                                                A_ub=np.array(A_ub),
                                                b_ub=np.array(b_ub),
                                                A_eq=np.array(A_eq),
                                                b_eq=np.array(b_eq),
                                                bounds=bounds,
                                                method='simplex').x)
        return scipy_sol[:-1]

    def multi_approx_br_to(self, profiles):
        def fun(x):
            return sum(p.exp_loss(x) for p in profiles)
        bnds = tuple([(0, 1) for t in self.M])
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        res = scipy.optimize.minimize(fun, util.gen_distr(len(self.M)),
                                      method='SLSQP', bounds=bnds,
                                      constraints=cons, tol=0.000001)
        return list(res.x)

    def mp_br_to(self, ap_tup, **kwargs):
        A_eq = [[1 for i in self.M] + [0]]
        b_eq = [self.resources]
        A_ub = []
        for t in self.M:
            terms = [self.game.values[t][ap_tup[0][0][0].id] * int(i != t)
                     for i in self.M]
            terms += [1]
            A_ub.append(terms)
        b_ub = [0 for i in range(len(A_ub))]
        bounds = [(0, 1) for i in self.M] + [(None, None)]
        c = [0 for i in self.M] + [0]
        for ap_t in list(ap_tup):
            for ap in list(ap_t):
                consts = ap[0].update_obj_fun(ap[1])
                c = [c[i] + consts[i] for i in range(len(c))]
        scipy_sol = list(scipy.optimize.linprog(c,
                                                A_ub=np.array(A_ub),
                                                b_ub=np.array(b_ub),
                                                A_eq=np.array(A_eq),
                                                b_eq=np.array(b_eq),
                                                bounds=bounds,
                                                method='simplex').x)
        return scipy_sol


class Attacker(Player):
    """
    The Attacker base class from which all the attacker inherit: it implements
    the best_respond method which is used by many types of adversaries.
    """

    def __init__(self, game, id, resources=1):
        """"
        Attributes

        feedbacks   list of targets dict with feedbacks for each turn
                    (if any)
        """
        super().__init__(game, id, resources)
        self.br = None
        self.last_br = None
        self.last_ol = None
        self.last_br_call = None
        self.closed_form_sol = True

    def finalize_init(self):
        super().finalize_init()
        if self.br is not None:
            self.br.finalize_init()

    def best_respond(self, strategies):
        """
        Compute the pure strategy that best respond to a given dict of
        defender strategies.
        In order to break ties, it selects the best choice for the defenders.
        """

        if not isinstance(strategies, dict):
            strategies = {0: strategies}

        targets = range(len(self.game.values))

        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]

        # (sum the probabilities of differents defenders)
        not_norm_coverage = sum(defenders_strategies)

        # normalize
        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)

        # compute the expected value of each target (v[t]*(1-c[t]))
        values = np.array([self.game.values[t][self.id] for t in targets])
        expected_payoffs = values * (np.ones(len(targets)) - coverage)
        expected_payoffs = [round(v, 3) for v in expected_payoffs]

        # play the argmax
        ordered_targets = sorted(targets,
                                 key=lambda t: expected_payoffs[t],
                                 reverse=True)[:]
        selected_targets = ordered_targets[:self.resources - 1]
        last_max = max([expected_payoffs[t] for t in targets
                        if t not in selected_targets])
        max_indexes = [i for i in targets if expected_payoffs[i] == last_max]
        # select the target which is the BEST for the defender (convention)
        # only 1st defender is taken into account
        d = self.game.defenders[0]
        best_for_defender = max(max_indexes, key=lambda x: -self.game.values[x][d])
        selected_targets.append(best_for_defender)
        return [int(t in selected_targets) for t in targets]

    def best_respond_mixed(self, strategies):
        """
        Compute the pure strategy that best respond to a given dict of
        defender strategies.
        it DOES randomize over indifferent maximum actions
        """
        targets = range(len(self.game.values))

        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]

        # (sum the probabilities of differents defenders)
        not_norm_coverage = sum(defenders_strategies)

        # normalize
        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)

        # compute the expected value of each target (v[t]*(1-c[t]))
        values = np.array([self.game.values[t][self.id] for t in targets])
        expected_payoffs = values * (np.ones(len(targets)) - coverage)
        expected_payoffs = [round(v, 3) for v in expected_payoffs]
        # play the argmax
        ordered_targets = sorted(targets,
                                 key=lambda t: expected_payoffs[t],
                                 reverse=True)[:]
        selected_targets = ordered_targets[:self.resources - 1]
        # randomize over the 'last resource'
        last_max = round(max([expected_payoffs[t] for t in targets
                              if t not in selected_targets]), 3)
        max_indexes = [i for i in targets if expected_payoffs[i] == last_max]
        selected_targets.append(np.random.choice(max_indexes))
        return [int(t in selected_targets) for t in targets]

#    def exp_loss(self, strategy_vec, **kwargs):
#        if isinstance(strategy_vec, dict):
#            return -sum([s_d *
#                         sum([s_a * sum(self.game.
#                                        get_player_payoffs(0, {0: [i], 1:[j]}))
#                              for j, s_a in enumerate(strategy_vec[1])
#                              if j != i])
#                         for i, s_d in enumerate(strategy_vec[0])])
#        else:
#            return self.exp_loss({0: strategy_vec,
#                                  1: self.compute_strategy(**kwargs)})


    def exp_loss(self, strategy_vec, **kwargs):
        if isinstance(strategy_vec, dict):
            if isinstance(self.game, game.PartialFeedbackGame):
                return -sum([s_d *
                            sum([s_a * (sum(self.game.
                                            get_player_payoffs(0, {0: [i], self.id:[j]}))
                                        if j != i else \
                                             -(self.game.values[i][self.id] * \
                                              (1 - self.game.observabilities.get(i))))
                                for j, s_a in enumerate(strategy_vec[self.id])
                                ])
                            for i, s_d in enumerate(strategy_vec[0])])
#                exp_loss = 0
#                for i, s_d in enumerate(strategy_vec[0]):
#                    for j, s_a in enumerate(strategy_vec[self.id]):
#                        guard = 1 if j != i else 1 - self.game.observabilities.get(i)
#                        payoffs = self.game.get_player_payoffs(0, {0: [i], self.id:[j]})
#                        exp_loss -= s_d * s_a * payoffs[j] * guard
#                return exp_loss
            else:
                return -sum([s_d *
                             sum([s_a * sum(self.game.
                                            get_player_payoffs(0, {0: [i], self.id:[j]}))
                                  for j, s_a in enumerate(strategy_vec[self.id])
                                  if j != i])
                             for i, s_d in enumerate(strategy_vec[0])])
        else:
            return self.exp_loss({0: strategy_vec,
                                  self.id: self.compute_strategy(**kwargs)})

    def best_response(self, **kwargs):
        """
        return the best response to this attacker, given the optional args.
        Default method returns the pure strategy which minimize the exp loss

        """
        m = min(self.M, key=lambda t: self.exp_loss(self.ps(t), **kwargs))
        return self.ps(m)

    def opt_loss(self, **kwargs):
        """
        kwargs in this case are useful only for learner/unknown parameters
        attackers

        """
        if not kwargs and self.last_ol is not None:
            return self.last_ol
        else:
            d_strat = self.best_response(**kwargs)
            a_strat = self.compute_strategy(**kwargs)
            s = {0: d_strat,
                 self.id: a_strat}
            self.last_ol = self.exp_loss(s)
            return self.last_ol

#    def loglk(self, old_loglk):
#        if old_loglk is None:
#            return None
#        elif isinstance(self.game, game.PartialFeedbackGame) and self.game.fake_target[-1] == 1:
#            o = self.game.perceived_target[-1]
#        else:
#            o = self.game.history[-1][1][0]
#        lkl = self.last_strategy[o]
#        if lkl == 0:
#            return None
#        else:
#            new_l = log(lkl)
#            return ((old_loglk * max(self.tau() - 1, 0) + new_l) /
#                    max(self.tau(), 1))

    def loglk(self, old_loglk):
        if old_loglk is None:
            return None
        if (not isinstance(self.game, game.PartialFeedbackGame)
            or self.game.fake_target[-1] == 0):
            o = self.game.history[-1][self.id][0]
            lkl = self.last_strategy[o]
        else:
            # if no feedback is received then we compute belief that defended target was not attacked
            def_targets = self.game.history[-1][self.game.defenders[0]]
            lkl = 1 - sum(self.last_strategy[t] for t in def_targets)
        if lkl == 0:
            return None
        new_l = log(lkl)
        return ((old_loglk * max(self.tau() - 1, 0) + new_l) /
                max(self.tau(), 1))

    def hloglk(self, old_loglk, hdict,
               history, ds_history):
        if old_loglk is None:
            return None
        o = history[-1][self.id]
        lkl = hdict["last_strategy"][o]
        if lkl == 0:
            return None
        else:
            new_l = log(lkl)
            return ((old_loglk * max(self.tau() - 1, 0) + new_l) /
                    max(self.tau(), 1))

    def hlearn(self, H, ds_history, hdict):
        return {}

    def get_attacker(self):
        att = deepcopy(self)
        att.game = self.game
        return att


def sample(distribution, items_number):
    selected_indexes = [e for e in range(len(distribution)) if distribution[e] == 1.0]
    dist = deepcopy(distribution)
    for e in selected_indexes:
        dist[e] = 0.0
    norm = items_number - len(selected_indexes)
    if norm != 0:
        dist = [p / norm for p in dist]
        selected_indexes= selected_indexes + list(np.random.choice(len(distribution),
                                                                    norm,
                                                                    p=dist,
                                                                    replace=False))
    return [e for e in selected_indexes]

class ObservingAttacker(Attacker):

    def best_respond(self, strategies):
        if not isinstance(strategies, dict):
            strategies = {0: strategies}
        targets = range(len(self.game.values))
        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]
        # (sum the probabilities of differents defenders)
        not_norm_coverage = sum(defenders_strategies)
        # normalize
        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)
        # compute the expected value of each target (v[t]*(1-o[t]*c[t])) if game has observabilities, otherwise as ususal
        values = np.array([self.game.values[t][self.id] for t in targets])
        if isinstance(self.game, game.PartialFeedbackGame):
            expected_payoffs = [values[t] * (1 - coverage[t]
                                * self.game.observabilities.get(t)) for t in targets]
        else:
            expected_payoffs = values * (np.ones(len(targets)) - coverage)
        expected_payoffs = [round(v, 3) for v in expected_payoffs]
        # play the argmax
        ordered_targets = sorted(targets,
                                 key=lambda t: expected_payoffs[t],
                                 reverse=True)[:]
        selected_targets = ordered_targets[:self.resources - 1]
        last_max = max([expected_payoffs[t] for t in targets
                        if t not in selected_targets])
        max_indexes = [i for i in targets if expected_payoffs[i] == last_max]
        # select the target which is the BEST for the defender (convention)
        # only 1st defender is taken into account
        d = self.game.defenders[0]
        best_for_defender = max(max_indexes, key=lambda x: -self.game.values[x][d])
        selected_targets.append(best_for_defender)
        return [int(t in selected_targets) for t in targets]

    def best_respond_mixed(self, strategies):
        """
        Compute the pure strategy that best respond to a given dict of
        defender strategies.
        it DOES randomize over indifferent maximum actions
        """
        targets = range(len(self.game.values))

        # compute total probability of being covered for each target (c[t])
        defenders_strategies = [np.array(strategies[d])
                                for d in self.game.defenders]

        # (sum the probabilities of differents defenders)
        not_norm_coverage = sum(defenders_strategies)

        # normalize
        coverage = not_norm_coverage / np.linalg.norm(not_norm_coverage,
                                                      ord=1)

        # compute the expected value of each target (v[t]*(1-c[t])) if game has observabilities, otherwise as ususal
        values = np.array([self.game.values[t][self.id] for t in targets])
        if isinstance(self.game, game.PartialFeedbackGame):
            expected_payoffs = [values[t] * (1 - coverage[t] *
                                self.game.observabilities.get(t)) for t in targets]
        else:
            expected_payoffs = values * (np.ones(len(targets)) - coverage)
        expected_payoffs = [round(v, 3) for v in expected_payoffs]
        # play the argmax
        ordered_targets = sorted(targets,
                                 key=lambda t: expected_payoffs[t],
                                 reverse=True)[:]
        selected_targets = ordered_targets[:self.resources - 1]
        # randomize over the 'last resource'
        last_max = round(max([expected_payoffs[t] for t in targets
                              if t not in selected_targets]), 3)
        max_indexes = [i for i in targets if expected_payoffs[i] == last_max]
        selected_targets.append(np.random.choice(max_indexes))
        return [int(t in selected_targets) for t in targets]
