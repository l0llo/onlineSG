from source.player import Player
from functools import reduce
import re
import numbers
from source.errors import NonHomogeneousTuplesError, TuplesWrongLenghtError
from copy import copy
import pickle
import json
from json import JSONEncoder
from copy import deepcopy
import numpy as np


class Game:
    """
    Base class for a generic Security Game (general sum) with a group of
    `attackers` and a group of `defenders`. It stores the history of plays
    and strategies of each players and the targets value matrix. Each security
    game application should have only one Game instance, but singleton design
    pattern has not been implemented in order to leave open the possibility
    of having multiple game, e.g. mock games for exploration purposes.
    
    A Game object can be created in several ways:

    * using its constructor, or using zero sum game one :meth:`zs_game`
    * from a pickled object :meth:`load`
    * parsing it from a configuration file (see :doc:`parsers`)

    :var values: tuple with a tuple for each target
                 with the values for each player
    :var time_horizon: the time horizon of the game
    :var players:   dict of players indexed by integers
    :var attackers: list of attackers indexes
    :var defenders: list of defenders indexes
    :var history:   list of dict for each turn: each one is made by the
                    moves of the players (each move is a tuple of choosen
                    targets indexes)
    :var strategy_history:  list of dict for each turn: each one is made by
                            the strategies of the players
    :var profiles:  list of attackers: the true attacker is of the same of one
        of them however it is NOT the same object. Profiles are "ghost" 
        players, in the sense that they evolve through time (some learn during
        the game) in order to mantain a correct model of the profile they
        represent. 
    """

    value_patterns = [re.compile(r"^\d+$"),
                      re.compile(r"^\(\d+( \d+)+\)$")]

    @classmethod
    def parse_value(cls, values, players_number):
        """
        returns a value tuple in order to instance a Game object
        """
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

        self.values = payoffs
        self.time_horizon = time_horizon
        self.players = dict()
        self.attackers = []
        self.defenders = []
        self.history = []
        self.strategy_history = []
        self.profiles = []


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

        for p in self.profiles:
            p.finalize_init()

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
                "Cannot compute utility for an index than does not exist: " +
                str(player_index) + " " + str(self.defenders)
            )

    def get_last_turn_payoffs(self, player_index):
        """
        returns the payoff list of the last turn given a player index
        """
        return self.get_player_payoffs(player_index, self.history[-1])

    def get_profiles_copies(self):
        """
        `DEPRECATED: use directly the actual profiles, without modifying them`

        returns a deep copy of the profiles 
        """
        profiles = deepcopy(self.profiles)
        for p in profiles:
            p.game = self  # copies need the real game!
        return profiles

    def is_finished(self):
        """
        True if the game has reached the time horizon
        """
        return len(self.history) >= self.time_horizon

    def dump(self, game_file):
        """
        saves a pickle of the game object in the **game_file** file
        """
        with open(game_file, mode='w+b') as file:
            pickle.dump(self, file)

    def dumpjson(self, jsonfile):
        """
        saves a json of the game object in the **jsonfile** file
        """
        with open(jsonfile, mode='w+') as f:
            f.write(json.dumps(self, cls=AutoJSONEncoder,
                               sort_keys=True, indent=4))

    def _json(self):
        return self.__dict__


def zs_game(values, time_horizon):
    """
    returns a zero sum game given the target values in **values**
    """
    payoffs = tuple((v, v) for v in values)
    return Game(payoffs, time_horizon)


def load(game_file):
    """
    loads a Game object from a picke in a **game_file** file.
    Game instances can be pickled using :meth:'Game.dump'
    """
    with open(game_file, mode='r+b') as file:
        game = pickle.load(file)
    return game


class AutoJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            return obj._json()
        except AttributeError:
            return JSONEncoder.default(self, obj)


def copy_game(g):
    """
    returns a shallow copy of thte game but with a copy of the history and
    strategy_history variables s.t. the original ones would not be modified.
    """
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
    """
    returns a toy example with 2 targets
    """
    example_targets = [1, 2]
    example_values = tuple((v, v) for v in example_targets)
    example_game = Game(example_values, 10)
    return deepcopy(example_game)


def example_game2():
    """
    returns a toy example with 2 targets and 2 players
    """
    g = example_game1()
    d = Player(g, 0, 1)
    a = Player(g, 1, 1)
    p = Player(g, 1, 1)
    g.set_players([d], [a], [p])
    return deepcopy(g)


if __name__ == '__main__':
    main()
