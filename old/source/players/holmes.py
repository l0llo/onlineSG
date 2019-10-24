import source.player as player
import source.players.base_defenders as base_defenders
import source.game as game
import source.players.attackers as attackers
import source.standard_player_parsers as spp
from copy import copy
from source.errors import NegativeProbabilityError
import re
import functools
import time
import logging
import source.belief
from copy import copy

logger = logging.getLogger(__name__)


class State:

    def __init__(self, b, r, p, g, a):
        self.b = b
        self.r = r
        self.p = p
        self.g = g
        self.a = a

    def __str__(self):
        return ('b: ' + str(self.b) +
                ' r: ' + str(self.r) +
                ' p: ' + str(self.p))

    def __repr__(self):
        return ('b: ' + str(self.b) +
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

    def __init__(self, game, id, resources, L, exploration=False):
        super().__init__(game, id, resources)
        self.profiles = []
        self.belief = None
        self.L = L
        self.arms = None
        self.learning = player.Learning.EXPERT
        self.t_strategies = None
        self.tree = None
        self.exploration = exploration

    def finalize_init(self):
        super().finalize_init()
        self.profiles = self.game.get_profiles_copies()
        for p in self.profiles:
            p.finalize_init()
        #self.belief = {k: 1 / (len(self.profiles)) for k in self.profiles}
        self.belief = source.belief.FrequentistBelief(self.profiles)
        self.arms = {k: k.get_best_responder() for k in self.profiles}

    def get_t_strategies(self):  # hardcoded for stackelberg!!!
        if self.exploration:
            if self.t_strategies is None:
                targets = list(range(len(self.game.values)))
                strategy = self.br_stackelberg()
                sel_targets = [t for t in targets if strategy[t] > 0]
                for epsilon in [0.0001, 0.001, 0.01, 0.1]:
                    t_strategies = []
                    while len([x for x in strategy if x < epsilon]):
                        epsilon /= 2
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
                if len([x for x in strategy if x < epsilon]):
                    raise NegativeProbabilityError([strategy, epsilon])
                self.t_strategies = t_strategies
            if len([k for k in self.profiles
                    if ((k.__class__.name == attackers.StackelbergAttacker.name) and
                        self.belief.pr[k] > 0)]):
                return self.t_strategies
        return []

    # def update_belief(self, o=None):
    #     """
    #     returns an updated belief, given an observation.
    #     If the observation is None, it uses the last game history
    #     """
    #     if o is None:
    #         o = self.game.history[-1][1][0]  # suppose 1 adversary, 1 resource
    #     update = {k: k.last_strategy[o] * self.belief[k]
    #               for k in self.profiles}
    #     eta = 1 / sum(update.values())
    #     update = {k: update[k] * eta for k in update}  # normalization
    #     return update

    def explore_state(self, state_node, depth, strategies=None):
        """
        dfs-limited tree generation
        """
        # print("exploring ", state_node, depth)
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
        #logger.info("exploring " + str(strategy) + " " + str(depth))
        #start_time = time.time()
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
                    for k in self.profiles:
                        k.game = g
                        a[k] = k.get_best_responder()
                        k.play_strategy()
                    update = {k: k.last_strategy[t] * state.b.pr[k]
                              for k in self.profiles}
                    g.history.append({0: [x],
                                      1: [t]})
                    # start_time = time.time()
                    for k in self.profiles:
                        a[k].receive_feedback(None)
                    # duration = time.time() - start_time
                    # logger.info(str(duration) + " " + str(strategy) + " " + str(depth))                        
                    p_t = sum(update.values())
                    if p_t == 0:    # Temporary solution, better to mark it as
                                    # UNREACHABLE state
                        b = None
                        #logger.info(str(s) + " " + str(x) + " " + str(t) + " unreachable")
                    else:
                        b = state.b.get_copy()
                        b.update(g.history[-1][1][0])
                    r = state.r + sum(g.get_last_turn_payoffs(0))
                    p = state.p * p_t * p_x
                    new_state = State(b, r, p, g, a)
                    s.branches[(x, t)] = State_Node(new_state, dict())
                if s.branches[(x, t)].state.p != 0:
                    if depth == 0:
                        exp_loss = sum([s.branches[(x, t)].state.b.pr[k] *
                                        k.opt_loss()
                                        for k in self.profiles])
                        regret = (-(s.branches[(x, t)].state.r) -
                                  exp_loss * self.L)
                       # duration = time.time() - start_time
                        #if duration > 0.1:
                        #logger.info(str(duration) + " " + str(strategy) + " " + str(depth))
                        s.exp_regret += (s.branches[(x, t)].state.p * regret)
                    else:
                        regret, min_s = self.explore_state(s.branches[(x, t)],
                                                           depth)
                        s.exp_regret += regret  # already weighted with p

    def get_br_strategies(self, arms):
        brs = []
        for p in self.profiles:
            if (p.__class__.name != attackers.StackelbergAttacker.name or
                not self.exploration):
                brs.append(tuple(arms[p].play_strategy()))
        return brs

    def compute_strategy(self):
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
        #self.belief = self.update_belief()
        self.belief.update(self.game.history[-1][1][0])

    def _json(self):
        d = super()._json()
        d.pop("belief", None)
        d.pop("profiles", None)
        d.pop("learning", None)
        d.pop("arms", None)
        return d

    def __str__(self):
        return "-".join([super().__str__()] +
                        [str(self.L)])
