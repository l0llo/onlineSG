import source.game as game
import source.player as player
import inspect
import pandas as pd
import source.players.attackers as attackers
import source.players.belief_max as bm
import source.players.fr as fr
import source.players.baseline as bl
from source.errors import *
import re
import collections
import source.players.base_defenders as bd
import source.players.defenders as defenders
import source.players.dmd as dmd
import source.players.bayesian as bd


class Parser:
    """
    Attributes
    df the pandas dataframe corresponding to the config file gives as input
    targets_headers          as below
    attackers_headers        as below
    defenders_headers        as below
    observability_headers    the relative dataframe header

    """

    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.targets_headers = []
        self.attackers_headers = []
        self.defenders_headers = []
        self.profile_headers = []
        self.observability_headers = []
        observability_pattern = re.compile(r'Obs(\d)+$')
        for h in self.df.columns:
            try:
                self.targets_headers.append(int(h))
            except ValueError:
                if "Attacker" in h:
                    self.attackers_headers.append(h)
                elif "Defender" in h:
                    self.defenders_headers.append(h)
                elif "Profile" in h:
                    self.profile_headers.append(h)
                elif observability_pattern.match(h):
                    self.observability_headers.append(h)
                elif h != "T" and h != "Name":
                    raise UnknownHeaderError
        observability_targets = [int(o[3:]) for o in self.observability_headers]
        if (observability_targets and
            collections.Counter(self.targets_headers) != collections.Counter(observability_targets)):
           raise TargetsAndObservabilitiesMismatchError

    def parse_row(self, index):
        """
        returns a game object from the row at the specified index of the config
        file. if the row as 'Obs' headers than a game with observabilities object
        will be returned: given observabilities shall be probability values and
        any non-probability given value will be assumed to be 1.
        """
        attacker_types = [self.df[a].iloc[index]
                          for a in self.attackers_headers
                          if isinstance(self.df[a].iloc[index], str)]
        defender_types = [self.df[d].iloc[index]
                          for d in self.defenders_headers
                          if isinstance(self.df[d].iloc[index], str)]
        profile_types = [self.df[d].iloc[index]
                         for d in self.profile_headers
                         if isinstance(self.df[d].iloc[index], str)]
        values = [self.df[str(t)].iloc[index]
                  for t in self.targets_headers]
        observabilities = dict()
        check_values = 0
        for o in self.observability_headers:
            try:
                    obs = round(float(self.df[o].iloc[index]), 3)
                    if 0 <= obs <= 1:
                        observabilities[int(o[3:])] = obs
                    else:
                        raise ValueError
            except ValueError:
                observabilities[int(o[3:])] = 1.0
        if all([observabilities.get(int(o[3:])) == 1.0 for o in self.observability_headers]): # No need to use observabilities if all = 1
            observabilities = None
        name = self.df["Name"].iloc[index]
        time_horizon = int(self.df["T"].iloc[index])
        player_number = len(attacker_types) + len(defender_types)
        try:
            if not observabilities:
                game = parse_game(values, player_number, time_horizon)
            else:
                game = parse_gamewithobservabilities(values, player_number, time_horizon, observabilities)
            defenders_ids = [parse_player(d, game, j)
                             for (j, d) in enumerate(defender_types)]
            attacker_ids = [parse_player(a, game, i + len(defenders_ids))
                            for (i, a) in enumerate(attacker_types)]
            profiles = [parse_player(a, game, 1)
                        for (i, a) in enumerate(profile_types)]
            game.set_players(defenders_ids, attacker_ids, profiles)
            return game, name
        except (UnparsableGameError, UnparsablePlayerError,
                TuplesWrongLenghtError) as e:
            raise RowError from e


def parse_player(player_type, game, id):
    """
    tries to parse the player_type calling the parse class method of all the
    classes of player module, and returns a Player or a subclass; otherwise
    raises an exception
    """
    players_classes = sum([get_classes(player),
                           get_classes(attackers),
                           get_classes(defenders),
                           get_classes(bd),
                           get_classes(bl),
                           get_classes(bm),
                           get_classes(fr),
                           get_classes(dmd),
                           get_classes(bd)], [])
    for c in players_classes:
        parsed = c.parse(player_type, game, id)
        if parsed:
            return parsed
    raise UnparsablePlayerError(player_type)


def parse_game(values, player_number, time_horizon):
    """
    tries to parse the values calling the parse class method of all the
    classes of game module, and then return a game; otherwise raises an
    exception
    """
    games_classes = [obj for name, obj in inspect.getmembers(game)
                     if inspect.isclass(obj) and
                     issubclass(obj, game.Game) and
                     hasattr(obj, 'parse_value')]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)

def parse_gamewithobservabilities(values, player_number, time_horizon, observabilities):
    """
    similar to the parse_game method, this method also tries to parse
    the observabilities of the game and then returns a game with
    observabilities
    """
    games_classes = [obj for name, obj in inspect.getmembers(game)
                     if inspect.isclass(obj) and
                     issubclass(obj, game.GameWithObservabilities) and
                     hasattr(obj, 'parse_value')]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon, observabilities)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)

def get_classes(module):
    return [c[1] for c in inspect.getmembers(module)
            if (inspect.isclass(c[1]) and
                c[1].__module__ == module.__name__ and
                hasattr(c[1], 'parse'))]
