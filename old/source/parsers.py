import source.game as gm
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
import source.players.frequentist as freqdef
import source.players.bayesian_approximator as bayapp
import source.players.partial_monitoring_forecaster as pmf


class Parser:
    """
    Attributes
    df the pandas dataframe corresponding to the config file gives as input
    targets_headers          as below
    attackers_headers        as below
    defenders_headers        the relative dataframe header

    """

    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.targets_headers = []
        self.attackers_headers = []
        self.defenders_headers = []
        self.profile_headers = []
        self.observability_headers = []
        self.feedback_prob_headers = []
        self.att_prof_prob_headers = []
        self.feed_type_header = None
        self.known_payoffs_header = None
        self.distinguishable_att_header = None
        self.att_prof_type_header = None
        observability_pattern = re.compile(r'Obs(\d)+$')
        feedback_prob_pattern = re.compile(r'Feed_prob(\d)+$')
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
                elif feedback_prob_pattern.match(h):
                    self.feedback_prob_headers.append(h)
                elif h == "Feed_type":
                    self.feed_type_header = h
                elif h == "Unknown_payoffs":
                    self.known_payoffs_header = h
                elif h == "Dist_att":
                    self.distinguishable_att_header = h
                elif h == "Att_prof_type":
                    self.att_prof_type_header = h
                elif "Att_prof_prob" in h:
                    self.att_prof_prob_headers.append(h)
                elif h != "T" and h != "Name":
                    raise UnknownHeaderError
        observability_targets = [int(o[3:]) for o in self.observability_headers]
        feedback_prob_targets = [int(fp[9:]) for fp in self.feedback_prob_headers]
        if (observability_targets and
            collections.Counter(self.targets_headers) != collections.Counter(observability_targets)):
           raise TargetsAndObservabilitiesMismatchError
        if (feedback_prob_targets and
            collections.Counter(self.targets_headers) != collections.Counter(feedback_prob_targets)):
           raise TargetsAndFeedbackProbabilitiesMismatchError
        if len(self.att_prof_prob_headers) > len(self.attackers_headers):
            raise AttackerProbabilitiesAndAttackerMismatchError

    def parse_row(self, index):
        """
        returns a game object from the row at the specified index of the config
        file. if the row as 'Obs' headers then a game with observabilities object
        will be returned: given observabilities shall be probability values and
        any non-probability given value will be assumed to be 1. Same for feedbacks.
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
        if self.att_prof_prob_headers:
            att_prof_prob = [self.df[p].iloc[index]
                                   for p in self.att_prof_prob_headers
                                   if isinstance(self.df[p].iloc[index], str)]
            att_prof_prob = [p.split('|') for p in att_prof_prob]
            att_prof_prob = [list(map(float, att_prof_prob[i])) for i in range(len(att_prof_prob))]
            for p in att_prof_prob:
                if round(sum(p), 3) != 1:
                    raise errors.NotAProbabilityError(p)
        observabilities = dict()
        feedback_prob = dict()
        feedback_type = None
        known_payoffs = True
        att_prof_type = "single"
        dist_att = True
        if self.feed_type_header:
            feedback_type = str(self.df[self.feed_type_header].iloc[index]).lower()
        if self.att_prof_type_header:
            att_prof_type = str(self.df[self.att_prof_type_header].iloc[index]).lower()
        if self.known_payoffs_header:
            known_payoffs = str(self.df[self.
                                known_payoffs_header].iloc[index]).lower() != "no"
        if self.distinguishable_att_header:
            dist_att = str(self.df[self.
                            distinguishable_att_header].iloc[index]).lower() != "no"
        for o in self.observability_headers:
            try:
                obs = round(float(self.df[o].iloc[index]), 3)
                if 0 <= obs <= 1:
                    observabilities[int(o[3:])] = obs
                else:
                    raise ValueError
            except ValueError:
                observabilities[int(o[3:])] = 1.0
        for fp in self.feedback_prob_headers:
            try:
                feed_prob = round(float(self.df[fp].iloc[index]), 3)
                if 0 <= feed_prob <= 1:
                    feedback_prob[int(fp[9:])] = feed_prob
                else:
                    raise ValueError
            except ValueError:
                feedback_prob[int(fp[9:])] = 1.0
#        observabilities = check_probability(self.observability_headers, self.df, index)
#        feedback_prob = check_probability(self.feedback_prob_headers, self.df, index)
        name = self.df["Name"].iloc[index]
        time_horizon = int(self.df["T"].iloc[index])
        player_number = len(attacker_types) + len(defender_types)
        try:
            if att_prof_type == "multi":
                game = parse_multi_profile_game(values, player_number,
                                time_horizon, observabilities, feedback_prob,
                                feedback_type, known_payoffs, dist_att)
            elif (self.observability_headers and any([observabilities.get(int(o[3:])) != 1.0
                  for o in self.observability_headers]) or
                  self.feedback_prob_headers and any([feedback_prob.get(int(fp[9:])) != 1.0
                  for fp in self.feedback_prob_headers]) or
                  feedback_type == "mab"):
                  game = parse_partial_feedback_game(values, player_number,
                                  time_horizon, observabilities, feedback_prob,
                                  feedback_type, known_payoffs, dist_att)
            else:
                game = parse_game(values, player_number, time_horizon,
                                known_payoffs, dist_att)
            defenders_ids = [parse_player(d, game, j)
                             for (j, d) in enumerate(defender_types)]
#            attacker_ids = [parse_player(a, game, i + len(defenders_ids))
#                            for (i, a) in enumerate(attacker_types)]
            attacker_ids = []
            for i, a in enumerate(attacker_types):
                match = re.search(r"-id(.+?)$",a)
                if match:
                    id_att = int(match.group(1))
                    attacker_types[i] = re.sub(r'-id(.+?)$', '',
                                            attacker_types[i])
                if att_prof_type != "multi":
                    attacker_ids.append(parse_player(attacker_types[i], game,
                                        id_att) if match
                                                else parse_player(a, game,
                                                        i + len(defenders_ids)))
                else:
                    attacker_ids.append(parse_multi_profile_player(attacker_types[i],
                                        game, id_att) if match
                                                      else parse_multi_profile_player(a,
                                                            game, i + len(defenders_ids)))
#            profiles = [parse_player(a, game, 1)
#                        for (i, a) in enumerate(profile_types)]

            if self.att_prof_prob_headers:
                if len(att_prof_prob) < len(attacker_ids):
                    l = len(attacker_ids) - len(att_prof_prob)
                    start = len(att_prof_prob)
                    for n in range(l):
                        att_prof_prob.append(util.gen_distr(
                                                len(att_prof_prob[start + n])))
                for n in range(len(attacker_ids)):
                    if len(attacker_ids[n]) != len(att_prof_prob[n]):
                            raise AttackerProbabilitiesAndAttackerMismatchError
            profiles = []
            for i, a in enumerate(profile_types):
                match = re.search(r"-id(.+?)$",a)
                if match:
                    id_prof = int(match.group(1))
                    profile_types[i] = re.sub(r'-id(.+?)$', '',
                                            profile_types[i])
                profiles.append(parse_player(profile_types[i], game,
                                id_prof) if match
                                else parse_player(a, game, 1))
            if type(game).__name__ == 'MultiProfileGame':
                game.set_players(defenders_ids, attacker_ids, profiles, att_prof_prob)
            else:
                game.set_players(defenders_ids, attacker_ids, profiles)
            return game, name
        except (UnparsableGameError, UnparsablePlayerError,
                TuplesWrongLenghtError) as e:
            raise RowError from e

#    def check_probability(probability_headers, df, index):
#        probability = dict()
#        for p in probability_headers:
#            try:
#                    prob = round(float(df[p].iloc[index]), 3)
#                    if 0 <= prob <= 1:
#                        probability[int(p[3:])] = prob
#                    else:
#                        raise ValueError
#            except ValueError:
#                probability[int(p[3:])] = 1.0
#        return probability

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
                           get_classes(freqdef),
                           get_classes(bayapp),
                           get_classes(pmf)], [])
    for c in players_classes:
        parsed = c.parse(player_type, game, id)
        if parsed:
            return parsed
    raise UnparsablePlayerError(player_type)

def parse_multi_profile_player(att_list, game, id):
    a_list = att_list.split('|')
    players_classes = sum([get_classes(player),
                           get_classes(attackers),
                           get_classes(defenders),
                           get_classes(bd),
                           get_classes(bl),
                           get_classes(bm),
                           get_classes(fr),
                           get_classes(dmd),
                           get_classes(freqdef),
                           get_classes(bayapp),
                           get_classes(pmf)], [])
    parsed_att = []
    for a in a_list:
        for c in players_classes:
            parsed = c.parse(a, game, id)
            if parsed:
                parsed_att.append(parsed)
    if len(parsed_att) == len(a_list):
        return parsed_att
    raise UnparsablePlayerError(player_type)

def parse_game(values, player_number, time_horizon, known_payoffs, dist_att):
    """
    tries to parse the values calling the parse class method of all the
    classes of game module, and then return a game; otherwise raises an
    exception
    """
    games_classes = [obj for name, obj in inspect.getmembers(gm)
                     if inspect.isclass(obj) and
                     name == "Game" and
                     hasattr(obj, 'parse_value')]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon, known_payoffs, dist_att)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)

def parse_partial_feedback_game(values, player_number, time_horizon,
                observabilities, feed_prob, feed_type, known_payoffs, dist_att):
    """
    similar to the parse_game method, this method also tries to parse
    the observabilities + feedbacks of the game and then returns a partial
    feedback game
    """
    games_classes = [obj for name, obj in inspect.getmembers(gm)
                     if inspect.isclass(obj) and
                     name == "PartialFeedbackGame" and
                     hasattr(obj, 'parse_value')]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon, observabilities, feed_prob,
                         feed_type, known_payoffs, dist_att)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)

def parse_multi_profile_game(values, player_number, time_horizon,
                observabilities, feed_prob, feed_type, known_payoffs, dist_att):
    """
    as parse_partial_feedback_game(), but for games with multi-profile attackers
    """
    games_classes = [obj for name, obj in inspect.getmembers(gm)
                     if inspect.isclass(obj) and
                     name == "MultiProfileGame" and
                     hasattr(obj, 'parse_value')]
    for c in games_classes:
        parsed_values = None
        try:
            parsed_values = c.parse_value(values, player_number)
            if parsed_values:
                return c(parsed_values, time_horizon, observabilities, feed_prob,
                         feed_type, known_payoffs, dist_att)
        except NonHomogeneousTuplesError as e:
            raise UnparsableGameError(values) from e
    raise UnparsableGameError(values)

def get_classes(module):
    return [c[1] for c in inspect.getmembers(module)
            if (inspect.isclass(c[1]) and
                c[1].__module__ == module.__name__ and
                hasattr(c[1], 'parse'))]
