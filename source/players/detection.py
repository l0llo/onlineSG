import source.player as player
import source.players.base_defenders as base_defenders
import source.game as game
import source.players.attackers as attackers
import source.standard_player_parsers as spp
from math import log, sqrt
from copy import copy, deepcopy
from functools import reduce
from collections import namedtuple
import enum
import re
import numpy as np
import time
import functools

Detection = enum.Enum('Detection', 'strategy_aware not_strategy_aware')
#State = namedtuple('State', ['b', 'r', 'p', 'g'])


class State:

    def __init__(self, b, r, p, g, a):
        self.b = b
        self.r = r
        self.p = p
        self.g = g
        self.a = a

    def __str__(self):
        return ('b: ' + str([v for k, v in self.b.items()]) +
                ' r: ' + str(self.r) +
                ' p: ' + str(self.p))

    def __repr__(self):
        return ('b: ' + str([v for k, v in self.b.items()]) +
                ' r: ' + str(self.r) +
                ' p: ' + str(self.p))


class State_Node:

    def __init__(self, state, branches):
        self.state = state
        self.branches = branches

    def __str__(self):
        return ('state: ' + str(self.state) +
                'branches: ' + str(self.branches))

    def __repr__(self):
        return ('state: ' + str(self.state) +
                'branches: ' + str(self.branches))

    def print_tree(self, level=0, end=None):
        print("\t" * level, 'state ', self.state)
        if level < end * 2:
            if len(self.branches) == 0:
                print("\t" * level, "pruned")
            else:
                print("\t" * level, 'branches:\n')
                for b, n in self.branches.items():
                    print("\t" * level, b)
                    n.print_tree(level + 1, end)


class Strategy_Node:

    def __init__(self, exp_regret, branches):
        self.exp_regret = exp_regret
        self.branches = branches

    def __str__(self):
        return ('exp_regret: ' + str(self.exp_regret) +
                'branches: ' + str(self.branches))

    def __repr__(self):
        return ('exp_regret: ' + str(self.exp_regret) +
                'branches: ' + str(self.branches))

    def print_tree(self, level=0, end=None):
        print("\t" * level, 'exp_regret ', self.exp_regret)
        print("\t" * level, 'branches:\n')    
        for b, n in self.branches.items():
            print("\t" * level, b)
            n.print_tree(level + 1, end)


class StrategyAwareDetector(base_defenders.StackelbergDefender):
    """
    only for non-history-based advesaries!
    """

    name = "sad"
    pattern = re.compile(r"^" + name + r"\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    def __init__(self, g, id, resources):
        super().__init__(g, id, resources)
        self.K2 = None
        self.K = None
        self.belief = None
        self.strategy_aware = None
        self.detection = None
        self.exp_defender = None
        self.learning = player.Learning.EXPERT
        self.arms = None

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = {k: k.last_strategy[o] * self.belief[k] for k in self.K}
        eta = 1 / sum(update.values())
        update = {k: update[k] * eta for k in update}  # normalization
        return update

    def find_t(self):
        """
        returns the t which maximizes the future belief of s.a.
        in case the prevision is true.
        """
        targets = list(range(len(self.game.values)))
        future_belief = dict()
        strategy = self.br_stackelberg()
        sel_targets = [t for t in targets if strategy[t] > 0]
        for t in sel_targets:
            self.strategy_aware.last_strategy = [int(t == i)
                                                 for i in targets]
            for k in self.K2:
                k.play_strategy()
            future_belief[t] = self.update_belief(t)
        sa_beliefs = [future_belief[t][self.strategy_aware]
                      for t in sel_targets]
        m = max(sa_beliefs)
        possible_t = [t for t in sel_targets
                      if future_belief[t][self.strategy_aware] == m]
        #target = np.random.choice(possible_t)
        target = possible_t[0]
        return target

    def compute_strategy(self):
        if self.tau == 0:
            self.K = deepcopy(self.game.profiles)
            for p in self.K:
                p.game = self.game  # copies need the real game!
            self.belief = {k: 1 / (len(self.K)) for k in self.K}
            self.strategy_aware = [k for k in self.K
                                   if (k.__class__ ==
                                       attackers.StackelbergAttacker)][0]
            self.K2 = copy(self.K)
            self.K2.remove(self.strategy_aware)
            experts = [k.get_best_responder() for k in self.K2]
            self.exp_defender = (base_defenders.
                                 ExpertDefender(self.game, self.id,
                                                self.resources, 1,
                                                'fpl', *experts))
            self.learning = player.Learning.EXPERT
            self.arms = [self.exp_defender]
        if self.detection is None:
            self.exp_defender.play_strategy()
            t = self.find_t()
            targets = list(range(len(self.game.values)))
            strategy = copy(self.br_stackelberg())
            sel_targets = [i for i in targets if strategy[i] > 0]
            epsilon = min(strategy[t] for t in sel_targets) / 100
            for i, s in enumerate(strategy):
                if i == t:
                    strategy[i] -= epsilon
                elif i in sel_targets:
                    strategy[i] += epsilon / (len(sel_targets) - 1)
            return strategy
        elif self.detection == Detection.strategy_aware:
            return self.br_stackelberg()
        elif self.detection == Detection.not_strategy_aware:
            return self.exp_defender.compute_strategy()

    def learn(self):
        if self.detection is None:
            for k in self.K:
                k.play_strategy()
            conditions = []
            if self.tau > 1:
                exceeded = []
                targets = list(range(len(self.game.values)))
                for i in self.K2:
                    #print(i.distribution)
                    conditions = []
                    for t in targets:
                        # how many times he has played adv_moves
                        n = len([h for h in self.game.history
                                 if h[1][0] == t])
                        sample_mean = n / (self.tau + 1)
                        #print(sample_mean)
                        p = i.last_strategy[t]  # STOCHASTIC ASSUMPTION!!!!!
                        conditions.append((abs(p - sample_mean) / 2) >
                                          sqrt(2 * log(self.tau + 1) / (self.tau + 1)))
                        #print(conditions[-1])
                    exceeded.append(reduce(lambda a, b: a or b, conditions))
                if reduce(lambda a, b: a and b, exceeded):
                    self.detection = Detection.strategy_aware
            self.belief = self.update_belief()
            if self.belief[self.strategy_aware] == 0:
                self.detection = Detection.not_strategy_aware

    def _json(self):
        d = super()._json()
        d.pop("belief", None)
        d.pop("K", None)
        d.pop("K2", None)
        d.pop("learning", None)
        d.pop("arms", None)
        return d


class HOLMES(base_defenders.StackelbergDefender):
    """
    Hints On L-Moves at Each Step
    Good old backward induction on the next l-steps
    """
    name = "holmes"
    pattern = re.compile(r"^" + name + r"\d+-\d+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_integers)

    def __init__(self, game, id, resources, L):
        super().__init__(game, id, resources)
        self.profiles = []
        self.belief = None
        self.L = L
        self.arms = None
        self.learning = player.Learning.MAB
        self.t_strategies = None
        self.tree = None

    def get_t_strategies(self):  # hardcoded for stackelberg!!!
        if self.t_strategies is None:
            targets = list(range(len(self.game.values)))
            strategy = self.br_stackelberg()
            sel_targets = [t for t in targets if strategy[t] > 0]
            for epsilon in [0.0001, 0.001, 0.01, 0.1]:
                t_strategies = []
                for t in sel_targets:
                    strategy = copy(self.br_stackelberg())
                    for i, s in enumerate(strategy):
                        if i == t:
                            strategy[i] -= epsilon
                        elif i in sel_targets:
                            strategy[i] += epsilon / (len(sel_targets) - 1)
                    t_strategies.append(tuple(strategy))
                # verify that they generate the correct br
                mock_att = player.Attacker(self.game, 1)
                br = [tuple(mock_att.best_respond({0: z, 1: None}))
                      for z in t_strategies]
                equalities = [(br[i] == tuple(int(j == t) for j in targets))
                              for i, t in enumerate(sel_targets)]
                if functools.reduce(lambda x, y: x and y, equalities):
                    break
            self.t_strategies = t_strategies
        return self.t_strategies

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = {k: k.last_strategy[o] * self.belief[k] for k in self.profiles}
        eta = 1 / sum(update.values())
        update = {k: update[k] * eta for k in update}  # normalization
        return update

    def explore_state(self, state_node, depth, strategies=None):  # df-limited tree generation
        #print("exploring ", state_node, depth)
        if strategies is None:
            strategies = (self.get_t_strategies() +
                          self.get_br_strategies(state_node.state.a))
        for s in strategies:
            if s not in state_node.branches:
                state_node.branches[s] = Strategy_Node(0, dict())
            self.explore_strategy(s, state_node.state,
                                  state_node.branches[s], depth)
        min_s = min(strategies,
                    key=lambda s: state_node.branches[s].exp_regret)
        return state_node.branches[min_s].exp_regret, min_s

    def explore_strategy(self, strategy, state, s, depth):
        #print("exploring ", strategy, depth)
        targets = list(range(len(self.game.values)))
        s.exp_regret = 0
        depth -= 1
        for x in [tar for tar in targets if strategy[tar] > 0]:
            p_x = strategy[x]
            for t in targets:
                if (x, t) not in s.branches:
                    g = game.copy_game(state.g)
                    # mock attacker strategy, cannot know the real one
                    g.strategy_history.append({0: strategy,
                                               1: [int(i == t)
                                                   for i in targets]})
                    a = dict()
                    # start_time = time.time()
                    for k in self.profiles:
                        k.game = g
                        a[k] = k.get_best_responder()
                        k.play_strategy()
                    update = {k: k.last_strategy[t] * state.b[k]
                              for k in self.profiles}
                    g.history.append({0: [x],
                                      1: [t]})
                    for k in self.profiles:
                        a[k].receive_feedback(None)
                    # duration = time.time() - start_time
                    # if duration > 0.1:
                    #     print(x, duration)
                    p_t = sum(update.values())
                    if p_t == 0:    # Temporary solution, better to mark it as
                                    # UNREACHABLE state
                        b = {k: 0 for k in update}
                    else:
                        b = {k: update[k] / p_t for k in update}
                    r = state.r + sum(g.get_last_turn_payoffs(0))
                    p = state.p * p_t * p_x
                    new_state = State(b, r, p, g, a)
                    s.branches[(x, t)] = State_Node(new_state, dict())
                if s.branches[(x, t)].state.p != 0:
                    if depth == 0:
                        exp_loss = sum([s.branches[(x, t)].state.b[k] *
                                        sum([k.exp_loss(z)
                                             for z
                                             in g.strategy_history[-self.L:]])
                                        for k in self.profiles])
                        regret = (-(s.branches[(x, t)].state.r) -
                                  exp_loss * self.L)
                        s.exp_regret += (s.branches[(x, t)].state.p * regret)
                    else:
                        regret, min_s = self.explore_state(s.branches[(x, t)],
                                                           depth)
                        s.exp_regret += regret  # already weighted with p

    def get_br_strategies(self, arms):
        return [tuple(arms[k].play_strategy()) for k in self.profiles
                if k.__class__.name != attackers.StackelbergAttacker.name]

    def compute_strategy(self):
        if self.tau == 0:
            self.profiles = deepcopy(self.game.profiles)
            self.belief = {k: 1 / (len(self.profiles)) for k in self.profiles}
            self.arms = {k: k.get_best_responder() for k in self.profiles}
        state = State(b=self.belief,
                      r=0,
                      p=1,
                      g=self.game,
                      a=self.arms)
        self.tree = State_Node(state, dict())
        br_strategies = self.get_br_strategies(self.arms)
        strategies = self.get_t_strategies() + br_strategies
        min_regret, min_s = self.explore_state(self.tree, self.L, strategies)
        if min_s in br_strategies:
            self.sel_arm = self.arms[self.profiles[br_strategies.index(min_s)]]
        else:
            self.sel_arm = None
        return min_s

    def learn(self):
        for p in self.profiles:
            p.game = self.game
            p.play_strategy()
        self.belief = self.update_belief()

    def _json(self):
        d = super()._json()
        d.pop("belief", None)
        d.pop("K", None)
        d.pop("learning", None)
        return d


class B2BW2W(base_defenders.StackelbergDefender):
    """
    "Bread to bread, wine to wine"
    """

    name = "b2bw2w"
    pattern = re.compile(r"^" + name + r"\d+$")
 #  pattern = re.compile(r"^" + name + r"\d+(@(.)+)+$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.parse1(cls, player_type, game, id, spp.parse_float)

    def __init__(self, game, id, resources):
        super().__init__(game, id, resources)
        self.profiles = None
        self.belief = None
        self.arms = None
        self.sel_arm = None
        self.learning = player.Learning.MAB

    def update_belief(self, o=None):
        """
        returns an updated belief, given an observation.
        If the observation is None, it uses the last game history
        """
        if o is None:
            o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
        update = {k: k.last_strategy[o] * self.belief[k]
                  for k in self.profiles}
        eta = 1 / sum(update.values())
        update = {k: update[k] * eta for k in update}  # normalization
        return update

    def compute_strategy(self):
        if self.tau == 0:
            self.profiles = deepcopy(self.game.profiles)
            for p in self.profiles:
                p.game = self.game  # copies need the real game!
            self.belief = {k: 1 / (len(self.profiles)) for k in self.profiles}
            self.arms = {k: k.get_best_responder() for k in self.profiles}
        chosen = player.sample([self.belief[k] for k in self.profiles], 1)[0]
        self.sel_arm = self.arms[self.profiles[chosen]]
        return self.sel_arm.play_strategy()

    def learn(self):
        # make our imagined adversary make a move 
        # pay attention to history-based ones!!! (use self.tau somehow?)
        for k in self.profiles:
            k.play_strategy()
        self.belief = self.update_belief()

    def _json(self):
        d = super()._json()
        d.pop("arms", None)
        d.pop("belief", None)
        d.pop("K", None)
        d.pop("learning", None)
        return d
