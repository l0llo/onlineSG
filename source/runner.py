from source.parsers import *
from source.environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join
from source.errors import UnknownHeaderError, RowError
import random


experiment = namedtuple('experiment', ['environment',
                                       'agent',
                                       'game',
                                       'results'])
aborted_experiment = namedtuple('aborted_experiment', ['error', 'row'])
batch = namedtuple('batch', ['name', 'parser', 'experiments'])
aborted_batch = namedtuple('aborted_batch', ['error', 'file'])


class Runner:

    def __init__(self, folder_path):
        files = [folder_path + '/' + f for f in listdir(folder_path)
                 if isfile(join(folder_path, f))]
        self.batches = []
        for f in files:
            try:
                b = batch(name=f, parser=Parser(f), experiments=[])
            except UnknownHeaderError as e:
                print("Something is wrong with the headers of ", f)
                self.batches.append(aborted_batch(e, f))
            except Exception as e:
                print("Something unexpected is happened with batch", f,
                      ":", e)
                self.batches.append(aborted_batch(e, f))
            else:
                self.batches.append(b)

    def run(self):
        for b in self.batches:
            run_batch(b)


def get_experiment(parser, index):
    game = parser.parse_row(index)
    environment = Environment(game, 0)
    agent = game.players[0]
    return experiment(game=game, environment=environment,
                      agent=agent, results=[])


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


def parse_batch(b):
    for row in range(len(b.parser.df)):
        try:
            i = get_experiment(b.parser, row)
            b.experiments.append(i)
        except RowError as e:
            print("Error in parsing row:", row, ": ", e)
            b.experiments.append(aborted_experiment(error=e, row=row))
        except Exception as e:
            print("Something unexpected is happened in experiment",
                  row, ": ", e)
            b.experiments.append(aborted_experiment(error=e, row=row))


def run_batch(b):
    for e in b:
        run_whole_experiment(e)


def init_seed():
    random.seed()
    seed = random.random()
    random.seed(seed)
    return seed
