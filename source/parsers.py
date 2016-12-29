import source.game as game
import source.player as player
import inspect
import pandas as pd
from source.errors import *


class Parser:
    """
    Attributes
    df                  the pandas dataframe corresponding to the
                        config file gives as input
    targets_headers     as below
    attackers_headers   as below
    defenders_headers   the relative dataframe header
    """

    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.targets_headers = []
        self.attackers_headers = []
        self.defenders_headers = []
        for h in self.df.columns:
            try:
                self.targets_headers.append(int(h))
            except ValueError:
                if "Attacker" in h:
                    self.attackers_headers.append(h)
                elif "Defender" in h:
                    self.defenders_headers.append(h)
                elif h != "T":
                    raise UnknownHeaderError

    def parse_row(self, index):
        """
        returns a game object from the row at the specified index of the config
        file.
        """
        attacker_types = [self.df[a].iloc[index]
                          for a in self.attackers_headers
                          if isinstance(self.df[a].iloc[index], str)]
        defender_types = [self.df[d].iloc[index]
                          for d in self.defenders_headers
                          if isinstance(self.df[d].iloc[index], str)]
        values = [self.df[str(t)].iloc[index]
                  for t in self.targets_headers]
        time_horizon = int(self.df["T"].iloc[index])
        player_number = len(attacker_types) + len(defender_types)
        try:
            game = parse_game(values, player_number, time_horizon)
            defenders = [parse_player(d, game, j)
                         for (j, d) in enumerate(defender_types)]
            attackers = [parse_player(a, game, i + len(defenders))
                         for (i, a) in enumerate(attacker_types)]
            game.set_players(defenders, attackers)
            return game
        except (UnparsableGameError, UnparsablePlayerError,
                TuplesWrongLenghtError) as e:
            raise RowError from e


def parse_player(player_type, game, id):
    """
    tries to parse the player_type calling the parse class method of all the
    classes of player module, and returns a Player or a subclass; otherwise
    raises an exception
    """
    players_classes = [obj for name, obj in inspect.getmembers(player)
                       if inspect.isclass(obj) and issubclass(obj, player.Player)]
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
                     if inspect.isclass(obj) and issubclass(obj, game.Game)]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)