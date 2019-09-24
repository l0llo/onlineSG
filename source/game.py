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
import source.util as util


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

    def __init__(self, payoffs, time_horizon, known_payoffs=True, dist_att=True):

        self.values = payoffs
        self.time_horizon = time_horizon
        self.players = dict()
        self.attackers = []
        self.defenders = []
        self.history = []
        self.strategy_history = []
        self.profiles = []
        self.known_payoffs = known_payoffs
        self.dist_att = dist_att


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
            raise TuplesWrongLenght
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

#    def shuffle_profiles(self):
#        self.profiles = np.random.shuffle(self.profiles)

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
            all_hit_targets = set(t for a in moves.keys() for t in moves[a]
                                  if t not in covered_targets
                                  and a not in self.defenders)
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

class PartialFeedbackGame(Game):

    def __init__(self, payoffs, time_horizon, observabilities=None,
                 feedback_prob=None, feedback_type=None, known_payoffs=True,
                 dist_att=True):
        super().__init__(payoffs, time_horizon, known_payoffs, dist_att)
        self.observabilities = dict()
        if type(observabilities) is dict and observabilities:
            self.observabilities = observabilities
        else:
            self.observabilities = util.gen_probabilities_with_len(len(payoffs))
        self.observation_history = []
        self.fake_target = []
        self.perceived_target = []
        self.feedback_prob = dict()
        if type(feedback_prob) is dict and feedback_prob:
            self.feedback_prob = feedback_prob
        else:
            self.feedback_prob = util.gen_probabilities_with_len(len(payoffs))
        if feedback_type == "mab":
            self.feedback_type = "mab"
        else:
            self.feedback_type = "full"

    def get_player_payoffs(self, player_index, moves, observations=None):
        """
        It returns the utility of a player given a dict
        of moves. Each move is a tuple of target indexes
        """
        if observations is not None:
            observed_targets = [k for k, v in observations.items() if v]
            covered_targets = set(t for d in self.defenders for t
                                  in list(set(moves[d])& set(observed_targets)))
        else:
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
        return self.get_player_payoffs(player_index, self.history[-1],
                                       self.observation_history[-1])

    def set_observabilities(self, observabilities):
        if len(observabilities) != len(self.values):
            print("Observabilities and targets have different lengths")
        else:
            self.observabilities = observabilities

    def set_feedback_prob(self, feedback_prob):
        if len(feedback_prob) != len(self.values):
            print("Feedback probabilities and targets have different lengths")
        else:
            self.feedback_prob = feedback_prob

    def sample_observation(self):
        """
        Samples observations for the moves made in a turn,
        according to the observabilities of the corresponding targets
        """
        observations = dict()
        observations = util.sample_probability(self.observabilities)
        self.observation_history.append(observations)

    def sample_feedback_prob(self):
        feedback_prob = {m: 0 for m in range(len(self.values))}
        if self.feedback_type == "mab":
            for d in self.defenders:
                for r in range(self.players[d].resources):
                    feedback_prob[self.history[-1][d][r]] = 1
        else:
            feedback_prob = util.sample_probability(self.feedback_prob)
        return feedback_prob

    def zs_partial_feedback_game(values, time_horizon):
        """
        returns a zero sum game given the target values in **values**
        """
        payoffs = tuple((v, v) for v in values)
        return PartialFeedbackGame(payoffs, time_horizon)

    def set_fake_target(self, feedback_prob):
        last_attacker_moves = [self.history[-1].get(a) for a in self.attackers]
        last_attacker_moves = list(util.flatten(last_attacker_moves))
        if any([feedback_prob.get(m) != 0 for m in last_attacker_moves]):
            self.fake_target.append(0)
        else:
            self.fake_target.append(1)
#        return self.fake_target[-1]

#    def set_perceived_target(self):
#        if self.fake_target[-1] != 1:
#            perceived_target = "-"
#        else:
#            perceived_target = np.random.choice([t for t in self.observabilities.keys()
#                                                 if not self.observation_history[-1].get(t)])
#        self.perceived_target.append(perceived_target)

#    def update_observabilities(self):
#        for t in range(len(self.values)):
#            obs = self.game.observabilities.get(t)
#            if t not in self.history[-1][0] & obs > 0.5:
#                self.game.observabilities[t] = obs * 0.95
#            elif obs <= 0.95:
#                self.game.observabilities[t] = obs / 0.95

class MultiProfileGame(PartialFeedbackGame):
    def __init__(self, payoffs, time_horizon, observabilities=None,
                 feedback_prob=None, feedback_type=None, known_payoffs=True,
                 dist_att=True):
        super().__init__(payoffs, time_horizon, observabilities, feedback_prob,
                         feedback_type, known_payoffs, dist_att)
        self.profiles_distribution = None
        self.profiles_history = []
        self.num_prof = 0

    def set_players(self, defenders, attackers, profiles, att_prob):
        id = attackers[0][0].id
        self.num_prof = len(attackers[0])
        for pl in attackers:
            self.attackers.append(pl[0].id)
        for p in defenders:
            self.players[p.id] = p
            self.defenders.append(p.id)
        np.random.shuffle(profiles)
        self.profiles = profiles

        for i in self.players:
            self.players[i].finalize_init()
        for p in self.profiles:
            p.finalize_init()

        self.profile_distribution = []
        for att_l, prob_l in zip(attackers, att_prob):
            self.profile_distribution.append([])
            for a, p in zip(att_l, prob_l):
                a.finalize_init()
                self.profile_distribution[-1].append((a, p))
            self.profile_distribution[-1] = tuple(self.profile_distribution[-1])
        self.profile_distribution = tuple(self.profile_distribution)

        self.set_attacker_profiles()

    def set_attacker_profiles(self):
        self.profiles_history.append([])
        for n in range(len(self.profile_distribution)):
            id = self.profile_distribution[n][0][0].id
            prob_list = [ap_pair[1] for ap_pair in self.profile_distribution[n]]
            sample = np.random.choice(len(prob_list), p=prob_list)
            sample_player = self.profile_distribution[n][sample][0]
            self.players[id] = sample_player
            self.profiles_history[-1].append(util.print_adv(sample_player))

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
