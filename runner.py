from parsers import *
from environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join


interaction = namedtuple('interaction', ['environment', 'agent', 'game']) #, 'status'])
batch = namedtuple('batch', ['name', 'parser', 'interactions'])


class Runner:

    def __init__(self, folder_path):
        files = [folder_path + '/' + f for f in listdir(folder_path)
                 if isfile(join(folder_path, f))]
        self.batches = [batch(name=f, parser=Parser(f), interactions=[])
                        for f in files]

    def run(self):
        for b in self.batches:
            run_batch(b)


def get_interaction(parser, index):
    game = parser.parse_row(index)
    environment = Environment(game, 0)
    agent = game.players[0]
    return interaction(game=game, environment=environment,
                       agent=agent)  #, status=False)


def run_interaction_cycle(i):
    strategy = i.agent.compute_strategy()
    i.environment.observe_strategy(strategy)
    realization = i.agent.sample_strategy()
    i.environment.observe_realization(realization)
    feedback = i.environment.feedback("expert")
    i.agent.receive_feedback(feedback)


def run_whole_interaction(i):
    while(not i.game.is_finished()):
        run_interaction_cycle(i)
    #i.status = True


def run_batch(b):
    for row in range(len(b.parser.df)):
        i = get_interaction(b.parser, row)
        b.interactions.append(i)
        run_whole_interaction(i)
