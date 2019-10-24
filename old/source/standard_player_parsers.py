"""
Collection of the most used 'parse' functions
"""
import source.errors as errors


def base_parse(cls, player_type, game, id):
    """
    name = ""
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    #__init__(self, game, id, resources=1):
    """
    if cls.pattern.match(player_type):
        args = [game, id] + [int(a) for a in
                             player_type.split(cls.name)[1].split("-")
                             if a != '']
        return cls(*args)
    else:
        return None


def parse_integers(cls, player_type, game, id):
    """
    name = ""
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    #__init__(self, game, id, resources=1):
    """
    args = [game, id] + [int(a) for a in
                         player_type.split(cls.name)[1].split("-")
                         if a != '']
    return args


def parse_float(cls, player_type, game, id):
    """
    name = ""
    pattern = re.compile(r"^" + name + "\d*$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.base_parse(cls, player_type, game, id)

    #__init__(self, game, id, resources=1):
    """
    arguments = [float(a) for a in
                 player_type.split(cls.name)[1].split("-")
                 if a != '']
    if len(arguments) > 0:
        arguments[0] = int(arguments[0])
    return [game, id] + arguments


def parse1(cls, player_type, game, id, parse_args):
    if cls.pattern.match(player_type):
        return cls(*parse_args(cls, player_type, game, id))
    else:
        return None


def stochastic_parse(cls, player_type, game, id):
    """
    name = ""
    pattern = re.compile(r"^" + name + r"(\d+(-\d+(\.\d+)?)*)?$")

    @classmethod
    def parse(cls, player_type, game, id):
        return spp.stochastic_parse(cls, player_type, game, id)
    #__init__(self, game, id, resources=1, *distribution):
    """
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


def profiles_parse(cls, player_type, game, id, adv_id, parsedefender):
    """
    name =
    pattern = re.compile(r"^"+ name + r"(@(.)+)+$")
    """
    import source.parsers as parsers

    if cls.pattern.match(player_type):
        substrings = player_type.split("@")
        arguments = parsedefender(cls, substrings[0], game, id)
        advs = []
        for s in substrings[1:]:
            try:
                a = parsers.parse_player(s, game, adv_id)
                advs.append(a)
            except errors.UnparsablePlayerError as e:
                raise errors.UnparsableProfile(e)
        arguments += advs
        return cls(*arguments)
