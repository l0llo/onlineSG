from source.parsers import *
from source.environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join


experiment = namedtuple('experiment', ['environment', 'agent', 'game'])  #, 'status'])
batch = namedtuple('batch', ['name', 'parser', 'experiments'])


class Runner:

    def __init__(self, folder_path):
        files = [folder_path + '/' + f for f in listdir(folder_path)
                 if isfile(join(folder_path, f))]
        self.batches = [batch(name=f, parser=Parser(f), experiments=[])
                        for f in files]

    def run(self):
        for b in self.batches:
            run_batch(b)


def get_experiment(parser, index):
    game = parser.parse_row(index)
    environment = Environment(game, 0)
    agent = game.players[0]
    return experiment(game=game, environment=environment,
                      agent=agent)  #, status=False)


def run_experiment_interaction(i):
    strategy = i.agent.compute_strategy()
    i.environment.observe_strategy(strategy)
    realization = i.agent.sample_strategy()
    i.environment.observe_realization(realization)
    feedback = i.environment.feedback("expert")
    i.agent.receive_feedback(feedback)


def run_whole_experiment(e):
    while(not e.game.is_finished()):
          run_experiment_interaction(e)
    #i.status = True


def run_batch(b):
    for row in range(len(b.parser.df)):
        i = get_experiment(b.parser, row)
        b.experiments.append(i)
        run_whole_experiment(i)
