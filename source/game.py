from source.player import Player
from functools import reduce
import re
import numbers
from source.errors import NonHomogeneousTuplesError, TuplesWrongLenghtError
from copy import copy
import pickle
import json
from json import JSONEncoder
from source.util import difficulty
from copy import deepcopy
import numpy as np


class Game:
    """
    """

    value_patterns = [re.compile(r"^\d+$"),
                      re.compile(r"^\(\d+( \d+)+\)$")]

    @classmethod
    def parse_value(cls, values, players_number):
        if reduce(lambda x, y: x and y, [isinstance(v, numbers.Number)
                                         for v in values]):
            return [[float(v) for p in range(players_number)]
                    for v in values]
        elif reduce(lambda x, y: x and y, [cls.value_patterns[0].match(v)
                                           for v in values]):
            return [[float(v) for p in range(players_number)]
                    for v in values]
        elif reduce(lambda x, y: x and y, [cls.value_patterns[1].match(v)
                                           for v in values]):
            value_tuples = [[int(i) for i in v.strip("()").split(' ')]
                            for v in values]
            for v in value_tuples:
                if len(v) != len(value_tuples[0]):
                    raise NonHomogeneousTuplesError
            return value_tuples
        else:
            return None

    def __init__(self, payoffs, time_horizon):

        #: tuple with a tuple for each target with the values for each
        #: player
        self.values = payoffs
        #:
        self.time_horizon = time_horizon
        #: dict of players indexed by integers
        self.players = dict()
        #: list of attackers' indexes
        self.attackers = []
        #: list of defenders' indexes
        self.defenders = []
        #: list of dict for each turn: each one is made by the
        #: moves of the players (each move is a tuple of choosen targets
        #: indexes)
        self.profiles = []
        self.history = []
        # self.difficulties = []
        self.strategy_history = []

    # def __str__(self):
    #     return ''.join(["<", self.__class__.__name__,
    #                     " values:", str(self.values),
    #                     " players", str(self.players),
    #                     " time_horizon:", str(self.time_horizon), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " values:", str(self.values),
                        " players", str(self.players),
                        " time_horizon:", str(self.time_horizon), ">"])

    def __str__(self):
        return ",".join([str(self.time_horizon)] +
                        [str(t[0]) for t in self.values] +
                        [str(self.players[p]) for p in self.players] +
                        [str(p) for p in self.profiles])

    def set_players(self, defenders, attackers, profiles):
        """
        run this method to the players to
        the game
        """

        #players = defenders + attackers
        # old_players_length = len(self.players)
        # for (i, p) in enumerate(players):
        #     self.players[i + old_players_length] = p
        # end_defenders = old_players_length + len(defenders)
        # end_attackers = end_defenders + len(attackers)
        # self.defenders.extend(list(range(old_players_length,
        #                                  end_defenders)))
        # self.attackers.extend(list(range(end_defenders,
        #                                  end_attackers)))
        for p in attackers:
            self.players[p.id] = p
            self.attackers.append(p.id)
        for p in defenders:
            self.players[p.id] = p
            self.defenders.append(p.id)
        if len(self.values[0]) != len(self.players):
            raise TuplesWrongLenghtError
        np.random.shuffle(profiles)
        self.profiles = profiles

        for i in self.players:
            self.players[i].finalize_init()
        # hardcoding for 1 resource
        # attacker = self.players[self.attackers[0]]
        # self.difficulties = [difficulty(attacker, p)
        #                      for p in profiles if p != attacker]

    def play_game(self):
        for t in range(self.time_horizon):
            self.play_turn()

    def get_player_payoffs(self, player_index, moves):
        """
        It returns the utility of a player given a dict
        of moves. Each move is a tuple of target indexes
        """
        covered_targets = set(t for d in self.defenders for t in moves[d])

        if player_index in self.attackers:
            hit_targets = set(t for t in moves[player_index]
                              if t not in covered_targets)
            return [v[player_index] * (i in hit_targets)
                    for (i, v) in enumerate(self.values)]
        elif player_index in self.defenders:
            all_hit_targets = set(t for a in self.attackers for t in moves[a]
                                  if t not in covered_targets)
            return [-(v[player_index]) * (i in all_hit_targets)
                    for (i, v) in enumerate(self.values)]
        else:
            raise Exception(
                "Cannot compute utility for an index than does not exist"
            )

    def get_last_turn_payoffs(self, player_index):
        return self.get_player_payoffs(player_index, self.history[-1])

    def get_profiles_copies(self):
        profiles = deepcopy(self.profiles)
        for p in profiles:
            p.game = self  # copies need the real game!
        return profiles

    def is_finished(self):
        return len(self.history) >= self.time_horizon

    def dump(self, game_file):
        with open(game_file, mode='w+b') as file:
            pickle.dump(self, file)

    def dumpjson(self, jsonfile):
        with open(jsonfile, mode='w+') as f:
            f.write(json.dumps(self, cls=AutoJSONEncoder,
                               sort_keys=True, indent=4))

    def _json(self):
        return self.__dict__


def zs_game(values, time_horizon):
    payoffs = tuple((v, v) for v in values)
    return Game(payoffs, time_horizon)


def load(game_file):
    with open(game_file, mode='r+b') as file:
        game = pickle.load(file)
    return game


class AutoJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            return obj._json()
        except AttributeError:
            return JSONEncoder.default(self, obj)

# def play_turn(self):

#     # Defenders compute strategies (it includes also computing rewards)
#     self.strategy_history.append(dict())
#     for d in self.defenders:
#         self.strategy_history[-1][d] = self.players[d].play_strategy()

#     # Attackers possibly observe and compute strategies
#     for a in self.attackers:
#         self.strategy_history[-1][a] = self.players[a].play_strategy()

#     # Players extract a sample from their strategies
#     self.history.append(dict())
#     for p in self.players:
#         self.history[-1][p] = self.players[p].sample_strategy()


def copy_game(g):
    game_copy = copy(g)
    game_copy.history = copy(g.history)
    game_copy.strategy_history = copy(g.strategy_history)
    return game_copy


def main():
    g = Game(((1, 1), (2, 2)), 3)
    d = Player(g, 0, 1)
    a = Player(g, 1, 1)
    g.players[0] = d
    g.players[1] = a
    g.defenders.append(0)
    g.attackers.append(1)
    g.play_game()
    print(g.history)


def example_game1():
    example_targets = [1, 2]
    example_values = tuple((v, v) for v in example_targets)
    example_game = Game(example_values, 10)
    return deepcopy(example_game)


def example_game2():
    g = example_game1()
    d = Player(g, 0, 1)
    a = Player(g, 1, 1)
    p = Player(g, 1, 1)
    g.set_players([d], [a], [p])
    return deepcopy(g)


if __name__ == '__main__':
    main()
