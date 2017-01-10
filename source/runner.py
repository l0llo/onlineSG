from source.environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join
from source.errors import UnknownHeaderError, RowError, FolderExistsError
from copy import deepcopy, copy
from shutil import copyfile
import source.parsers as parsers
import random
import os
import pandas as pd

AbortedExperiment = namedtuple('AbortedExperiment', ['error', 'seed'])
AbortedConfiguration = namedtuple('AbortedConfiguration', ['error', 'row'])
AbortedBatch = namedtuple('AbortedBatch', ['error', 'file'])


class Runner:

    def __init__(self, batches_folder_path,
                 results_folder_path, print_results=True):
        self.print_results = print_results
        files = [batches_folder_path + '/' + f
                 for f in listdir(batches_folder_path)
                 if isfile(join(batches_folder_path, f))]
        self.results_folder_path = results_folder_path
        if os.path.exists(results_folder_path):
            raise FolderExistsError(results_folder_path)
        else:
            os.makedirs(results_folder_path)
        self.batches = []
        for f in files:
            try:
                b = Batch(f, results_folder_path, self.print_results)
            except UnknownHeaderError as e:
                print("Something is wrong with the headers of ", f)
                self.batches.append(AbortedBatch(e, f))
            except Exception as e:
                print("Something unexpected is happened with batch", f,
                      ":", e)
                self.batches.append(AbortedBatch(e, f))
            else:
                self.batches.append(b)

    def run(self):
        for b in self.batches:
            if isinstance(b, Batch):
                b.parse_batch()
                b.run()
                """
                at this point if we have an high number of experiment
                to be done, it could be useful to eliminate the batch or
                its attributes to let the garbage collector eliminate them
                and therefore free some memory.
                A boolen attribute could be added to regulate it.
                """

    def __str__(self):
        str1 = ''.join(["<", self.__class__.__name__, " batches:"])
        str2 = ''.join(["\n" + str(b) for b in self.batches]) + ">"
        return str1 + str2

    def __repr__(self):
        str1 = ''.join(["<", self.__class__.__name__, " batches:"])
        str2 = ''.join(["\n" + str(b) for b in self.batches]) + ">"
        return str1 + str2


class Batch:

    def __init__(self, name, results_folder_path, print_results=True):
        self.print_results = print_results
        self.name = name
        self.parser = parsers.Parser(name)
        self.configurations = []
        file_name = self.name.split("/")[-1].split(".")[0]
        self.results_folder_path = results_folder_path + "/" + file_name
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path)
        copyfile(name, self.results_folder_path + "/batch.csv")

    def parse_batch(self):
        for row in range(len(self.parser.df)):
            try:
                game = self.parser.parse_row(row)
                conf_path = self.results_folder_path + "/" + str(row)
                c = Configuration(game, conf_path, self.print_results)
                self.configurations.append(c)
            except RowError as e:
                print("Error in parsing row:", row, ": ", e)
                self.configurations.append(AbortedConfiguration(error=e, row=row))
            except Exception as e:
                print("Something unexpected is happened in configuration ",
                      row, ": ", e)
                self.configurations.append(AbortedConfiguration(error=e, row=row))

    def run(self):
        for c in self.configurations:
            if isinstance(c, Configuration):
                c.run_an_experiment()


    def __str__(self):
        str1 = ''.join(["<", self.__class__.__name__, " configurations:"])
        str2 = ''.join(["\n" + str(b) for b in self.configurations]) + ">"
        return str1 + str2

    def __repr__(self):
        str1 = ''.join(["<", self.__class__.__name__, " configurations:"])
        str2 = ''.join(["\n" + str(b) for b in self.configurations]) + ">"
        return str1 + str2


def init_seed():
    random.seed()
    seed = random.random()
    random.seed(seed)
    return seed


class Configuration:

    def __init__(self, game, results_folder_path, print_results=True):
        self.print_results = print_results
        self.game = game
        self.experiments = []
        self.results_folder_path = results_folder_path
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path)
        pickle_file = self.results_folder_path + "/game"
        json_file = self.results_folder_path + "/json.txt"
        self.game.dump(pickle_file)
        self.game.dumpjson(json_file)
        self.stats = {}

    def compute_stats(self):

        self.stats['avg_total_rewards'] = sum([e.stats['total_rewards']
                                               for e in self.experiments
                                               if isinstance(e, Experiment)])
        self.stats['avg_weak_regret'] = sum([e.stats['weak_regret']
                                             for e in self.experiments
                                             if isinstance(e, Experiment)])
        exp_number = len([e for e in self.experiments 
                          if isinstance(e, Experiment)])
        if exp_number:
            self.stats['avg_total_rewards'] /= exp_number
            self.stats['avg_weak_regret'] /= exp_number

    def run_an_experiment(self, seed=None):
        if not seed:
            seed = init_seed()
        game = deepcopy(self.game)
        experiment = Experiment(game, seed)
        try:
            experiment.run()
            if self.print_results:
                experiment.save_results(self.results_folder_path)
        except Exception as e:
            print("Something has gone wrong with the experiment: ", e)
            self.experiments.append(AbortedExperiment(e, seed))
        else:
            self.experiments.append(experiment)
        self.compute_stats()

    def results_of(self, index, start=None, end=None):
        return self.experiments[index].results(self.results_folder_path,
                                               start, end)

    def all_results(self, start=None, end=None):
        return [e.results(self.results_folder_path, start, end)
                for e in experiments]

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " game:", str(self.game),
                        " experiments:", str(self.experiments),
                        " stats:", str(self.stats), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " game:", str(self.game),
                        " experiments:", str(self.experiments),
                        " stats:", str(self.stats), ">"])


class Experiment:

    def __init__(self, game, seed=None):
        if not seed:
            seed = init_seed()
        else:
            random.seed(seed)
        self.game = game
        self.environment = Environment(game, 0)
        self.agent = game.players[0]
        self.seed = seed
        self.stats = None

    def run_interaction(self):
        strategy = self.agent.compute_strategy()
        self.environment.observe_strategy(strategy)
        realization = self.agent.sample_strategy()
        self.environment.observe_realization(realization)
        feedback = self.environment.feedback("expert")
        self.agent.receive_feedback(feedback)

    def run(self):
        while(not self.game.is_finished()):
            self.run_interaction()
        self.compute_stats()

    def save_results(self, folder):
        df = pd.DataFrame()
        for p in self.game.players:
            key = self.game.players[p].__class__.name + "-" + str(p)
            df[key] = [([round(s, 2) for s in self.game.strategy_history[i][p]], h[p])
                       for (i, h) in enumerate(self.game.history)]
        targets = list(range(len(self.game.values)))

        for t in targets:
            key = "feedback target " + str(t)
            df[key] = [f[t] for f in self.agent.feedbacks]
        df["total"] = [f['total'] for f in self.agent.feedbacks]
        df.to_csv(folder + "/" + str(self.seed))
        f = open(folder + "/seeds.txt", "a")
        f.write(str(self.seed) + "\n")
        f.close()

    def results(self, folder, start=None, end=None):
        return pd.read_csv(folder + "/" + str(self.seed), index_col=0).iloc[start:end]

    def total_rewards(self):
        return sum([f['total'] for f in self.agent.feedbacks])

    def fixed_action_reward(self, a):
        reward = 0
        for h in self.game.history:
            moves = copy(h)
            moves[0] = [a]
            reward += sum(self.game.get_player_payoffs(0, moves))
        return reward

    def best_action_in_hindsight(self):
        actions = list(range(len(self.game.values)))
        rewards = [self.fixed_action_reward(a) for a in actions]
        best = max(actions, key=lambda x: rewards[x])
        return (best, rewards[best])

    def weak_regret(self):
        return self.best_action_in_hindsight()[1] - self.total_rewards()

    def compute_stats(self):
        self.stats = {}
        self.stats['total_rewards'] = self.total_rewards()
        self.stats['best_action'] = self.best_action_in_hindsight()[0]
        self.stats['weak_regret'] = self.weak_regret()

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed),
                        " stats:", str(self.stats), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed),
                        " stats:", str(self.stats), ">"])
