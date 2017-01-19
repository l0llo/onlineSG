"""
ROADMAP

1 defender and 1 attacker, with one resource each

- Possible types:
    - Known Stochastic      V
    - Unknown Stochastic    V
    - Stackelberg           V
    - Fictitious Player     ? (stackelberg?)
    - Quantal Response      -
    - Adversary             -

- 2 types:
    -1 Known Stochastic or Stackelberg           V
    -2 Unknown Stochastic or Stackelberg         ~
    -3 Known Stochastic or Fictitious Player     ? 
    -4 Unknown Stochastic or Fictitious Player   -
    -5 Stackelberg or Fictitious Player          ?
    -6 Known Stochastic or Quantal Response      -
    -7 Unknown Stochastic or Quantal Response    -
    -8 Stackelberg or Quantal Response           -
    -9 Fictitious Player or Quantal response     -
    -10 Known Stochastic or Adversary            -
    -11 Unknown Stochastic or Adversary          -
    -12 Stackelberg or Adversary                 -
    -13 Fictitious Player or Adversary           -
    -14 Quantal Response or Adversary            -

Then we can try to distinguish among sets of types:

- discriminate among a set by eliminating one type at time using the known
  bounds
- e.g. {KS, Sta, QR}: 2 bounds for KS, see which is passed before
- if it is passed => discard KS!
- otherwise => it could be KS


Extend the 2-types identification with multiple resources

Extend the multiple-types

Extend the number of attackers (not coordinated)

Extend the number of defenders


"""

import source.player as player
import source.players.attackers as attackers
import source.players.base_defenders as base_defenders
import source.errors as errors
from math import log, sqrt, exp
from copy import deepcopy
import enum
import re
import gurobipy
import numpy as np

State = enum.Enum('State', 'exploring stochastic stackelberg')


class StUDefender(player.Defender):
    """
    This defender is able to distinguish between a uniform
    or a stackelberg attacker and best respond accordingly
    """
    name = "stu_defender"
    pattern = re.compile(r"^" + name + "\d*$")

    def __init__(self, game, id, resources=1, confidence=0.9):
        super().__init__(game, id, resources)
        self.confidence = confidence
        self.mock_stackelberg = attackers.StackelbergAttacker(self.game, 1)
        self.belief = {'uniform': 1,
                       'stackelberg': 0}
        self.br_stackelberg_strategy = None

    def compute_strategy(self):
        if len(self.game.history) == 1 or self.belief['stackelberg']:
            last_move = self.game.history[-1][1][0]
            mock_move = [i for (i, s) in enumerate(self.mock_stackelberg.compute_strategy())
                         if s][0]
            if last_move == mock_move:
                self.belief['uniform'] = self.belief['uniform'] * (1 / len(self.game.values))
                self.belief['stackelberg'] = 1 - self.belief['uniform']
            else:
                self.belief['uniform'] = 1
                self.belief['stackelberg'] = 0

        if self.belief['stackelberg']:
            return self.br_stackelberg()  # minimax in two players game
        else:
            return self.br_uniform()  # highest value action

    def br_stackelberg(self):
        if not self.br_stackelberg_strategy:
            m = gurobipy.Model("SSG")
            targets = list(range(len(self.game.values)))
            strategy = []
            for t in targets:
                strategy.append(m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name="x" + str(t)))
            v = m.addVar(lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name="v")
            m.setObjective(v, gurobipy.GRB.MAXIMIZE)
            for t in targets:
                terms = [-self.game.values[t][self.id] * strategy[i]
                         for i in targets if i != t]
                m.addConstr(sum(terms) - v >= 0, "c" + str(t))
            m.addConstr(sum(strategy) == 1, "c" + str(len(targets)))
            m.params.outputflag = 0
            m.optimize()
            self.br_stackelberg_strategy = [float(s.x) for s in strategy]
        return self.br_stackelberg_strategy


class FABULOUS(base_defenders.StackelbergDefender,
               base_defenders.KnownStochasticDefender):
    """
    Adapt When Everybody is Stochastic, Otherwise Move to Stackelberg
    with Upper Confidence Bound where p=t^(-4)

    Defender detection of Stackelberg-Known Stochastic
    """

    name = "fabulous"
    pattern = re.compile(r"^" + name + r"((\d+(\.\d+)?)+(-(\d+(\.\d+)?)+)+)?$")

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

    def __init__(self, game, id, resources=1, *distribution):
        player.Defender.__init__(self, game, id, resources)
        self.distribution = distribution
        self.maxmin = None
        self.stochastic_reward = None
        self.br_stackelberg_strategy = None
        self.br_stochastic_strategy = None



        # Initialization
        self.norm_const = 1  # has to be initialized late
        self.br_stackelberg()
        self.br_stochastic()
        self.stochastic_loss = - self.stochastic_reward
        self.loss_sta_sto = None
        self.state = State.exploring

    def compute_strategy(self):
        # if it is the first round then br to stackelberg
        t = len(self.game.history)
        if t == 0:
            self.norm_const = max([v[self.id] for v in self.game.values])
            # mock_stackelberg = player.Attacker(self.game, 1)
        if t < 2:
            return self.br_stochastic()
        else:
            def_last_moves = set(self.game.history[-1][0])  # hardcoded for now
            att_last_moves = set(self.game.history[-1][1])
            if self.state == State.exploring:
                if def_last_moves.intersection(att_last_moves):
                    self.state = State.stochastic
                    return self.br_stochastic()
                avg_loss = - sum([f['total'] for f in self.feedbacks]) / t
                delta_t = 1 / (t * t)
                bound = self.norm_const * sqrt(-log(delta_t) / (t))
                if avg_loss - self.stochastic_loss <= bound:
                    return self.br_stochastic()
                else:
                    #print("bound: ", bound, "diff: ", avg_loss - self.stochastic_loss)
                    self.state = State.stackelberg
                    return self.br_stackelberg()
            elif self.state == State.stochastic:
                return self.br_stochastic()
            elif self.state == State.stackelberg:
                return self.br_stackelberg()

    def _json(self):
        self_copy = deepcopy(self)
        d = self_copy.__dict__
        d.pop("game", None)
        d["state"] = d["state"].name
        d["class_name"] = self.__class__.__name__
        return d
