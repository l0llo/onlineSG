from source.player import Player
from functools import reduce
import re
import numbers
from source.errors import NonHomogeneousTuplesError, TuplesWrongLenghtError
import pickle
import json
from json import JSONEncoder


class Game:
    """
    """

    value_patterns = [re.compile(r"^\d+$"),
                      re.compile(r"^\(\d+( \d+)+\)$")]

    @classmethod
    def parse_value(cls, values, players_number):
        if reduce(lambda x, y: x and y, [isinstance(v, numbers.Number)
                                         for v in values]):
            return [[v for p in range(players_number)]
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
        self.history = []
        self.strategy_history = []

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " values:", str(self.values),
                        " players", str(self.players),
                        " time_horizon:", str(self.time_horizon), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " values:", str(self.values),
                        " players", str(self.players),
                        " time_horizon:", str(self.time_horizon), ">"])

    def set_players(self, defenders, attackers):
        """
        run this method to add new players to
        the game
        """
        old_players_length = len(self.players)
        players = defenders + attackers
        for (i, p) in enumerate(players):
            self.players[i + old_players_length] = p
        end_defenders = old_players_length + len(defenders)
        end_attackers = end_defenders + len(attackers)
        self.defenders.extend(list(range(old_players_length,
                                         end_defenders)))
        self.attackers.extend(list(range(end_defenders,
                                         end_attackers)))
        if len(self.values[0]) != len(players):
            raise TuplesWrongLenghtError

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
#         self.strategy_history[-1][d] = self.players[d].compute_strategy()

#     # Attackers possibly observe and compute strategies
#     for a in self.attackers:
#         self.strategy_history[-1][a] = self.players[a].compute_strategy()

#     # Players extract a sample from their strategies
#     self.history.append(dict())
#     for p in self.players:
#         self.history[-1][p] = self.players[p].sample_strategy()


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


if __name__ == '__main__':
    main()
