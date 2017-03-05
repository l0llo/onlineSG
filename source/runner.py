from source.environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join
from source.errors import UnknownHeaderError, RowError, FolderExistsError
from copy import deepcopy, copy
from shutil import copyfile
import source.parsers as parsers
import source.game as game
import os
import re
import pandas as pd
import traceback
import numpy as np
import shutil
from source.util import exp_regret, actual_regret, exp_loss, log_progress
import concurrent.futures


AbortedExperiment = namedtuple('AbortedExperiment', ['error', 'info', 'seed'])
AbortedConfiguration = namedtuple('AbortedConfiguration', ['error', 'info', 'row'])
AbortedBatch = namedtuple('AbortedBatch', ['error', 'file'])


class Runner:

    def __init__(self, batches_folder_path,
                 results_folder_path, print_results=True):
        self.print_results = print_results
        files = [batches_folder_path + '/' + f
                 for f in listdir(batches_folder_path)
                 if isfile(join(batches_folder_path, f))]
        self.results_folder_path = results_folder_path
        # if os.path.exists(results_folder_path):
        #     raise FolderExistsError(results_folder_path)
        # else:
        #     os.makedirs(results_folder_path)
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

    def run(self, n=1):
        for b in self.batches:
            if isinstance(b, Batch):
                b.parse_batch()
                b.run(n)
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
        # file_name = re.split("[0-9]+", self.name.split("/")[-1].
        #                      split(".")[0], 1)[0]
        file_name = self.name.split("/")[-1].split(".")[0]
        self.results_folder_path = results_folder_path + "/" + file_name
        if not os.path.exists(self.results_folder_path):
            os.makedirs(self.results_folder_path)
        copyfile(name, self.results_folder_path + "/batch.csv")

    def parse_batch(self):
        for row in range(len(self.parser.df)):
            try:
                g, name = self.parser.parse_row(row)
                c = Configuration(g, self.results_folder_path, name, self.print_results)
                self.configurations.append(c)
            except RowError as e:
                print("Error in parsing row:", row, ": ", e)
                self.configurations.append(AbortedConfiguration(e,
                                                                traceback.format_exc(),
                                                                row))
            except Exception as e:
                print("Something unexpected is happened in configuration ",
                      row, ": ", e)
                self.configurations.append(AbortedConfiguration(e,
                                                                traceback.format_exc(),
                                                                row))

    def run(self, n=1, show_progress=False, workers=None):
        if show_progress:
            for c in log_progress(self.configurations):
                if isinstance(c, Configuration):
                    c.run(n, workers)
        else:
            for c in self.configurations:
                if isinstance(c, Configuration):
                    c.run(n, workers)

    def __str__(self):
        str1 = ''.join(["<", self.__class__.__name__, " configurations:"])
        str2 = ''.join(["\n" + str(b) for b in self.configurations]) + ">"
        return str1 + str2

    def __repr__(self):
        str1 = ''.join(["<", self.__class__.__name__, " configurations:"])
        str2 = ''.join(["\n" + str(b) for b in self.configurations]) + ">"
        return str1 + str2


def init_seed():
    np.random.seed()
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    return seed


class Configuration:

    def __init__(self, g, folder, name, print_results=True):
        self.print_results = print_results
        self.game = g
        self.experiments = []
        self.name = name
        self.results_folder_path = folder + "/" + name
        if os.path.exists(self.results_folder_path + "/experiments"):
            raise FolderExistsError(self.results_folder_path)
        else:
            os.makedirs(self.results_folder_path + "/experiments")
        pickle_file = self.results_folder_path + "/game"
        json_file = self.results_folder_path + "/json.txt"
        self.game.dump(pickle_file)
        self.game.dumpjson(json_file)
        self.stats = {}

    def compute_stats(self):

        self.stats['avg_total_rewards'] = sum([e.stats['total_rewards']
                                               for e in self.experiments
                                               if isinstance(e, Experiment)])
        self.stats['avg_exp_regret'] = sum([e.stats['exp_regret']
                                            for e in self.experiments
                                            if isinstance(e, Experiment)])
        exp_number = len([e for e in self.experiments
                          if isinstance(e, Experiment)])
        if exp_number:
            self.stats['avg_total_rewards'] /= exp_number
            self.stats['avg_exp_regret'] /= exp_number

    def run_an_experiment(self, seed=None):
        if not seed:
            seed = init_seed()
        g = deepcopy(self.game)
        experiment = Experiment(g, seed)
        try:
            experiment.run()
            if self.print_results:
                experiment.save_results(self.results_folder_path + "/experiments")
        except Exception as e:
            print("Something has gone wrong with the experiment: ", e)
            return AbortedExperiment(e, traceback.format_exc(), seed)
        else:
            return experiment

    def run(self, n=1, workers=None):
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            futures = []
            for i in range(n):
                futures.append(executor.submit(Configuration.run_an_experiment,
                                               self))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            self.experiments.append(result)
        self.compute_stats()
        self.print_stats_file()

    def print_stats_file(self):
        df = pd.DataFrame()
        df["exp_loss"] = (sum([np.array(e.exp_loss)
                               for e in self.experiments
                               if isinstance(e, Experiment)]) /
                          len(self.experiments))
        df["actual_regret"] = (sum([np.array(e.actual_regret)
                                    for e in self.experiments
                                    if isinstance(e, Experiment)]) /
                               len(self.experiments))
        df["exp_regret"] = (sum([np.array(e.exp_regret)
                                 for e in self.experiments
                                 if isinstance(e, Experiment)]) /
                            len(self.experiments))
        df.to_csv(self.results_folder_path + "/stats.csv")

    def results_of(self, index, start=None, end=None):
        return self.experiments[index].results(self.results_folder_path +
                                               "/experiments",
                                               start, end)

    def all_results(self, start=None, end=None):
        return [e.results(self.results_folder_path +
                          "/experiments", start, end)
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

    def __init__(self, g, seed=None):
        if not seed:
            seed = init_seed()
        else:
            np.random.seed(seed)
        self.game = g
        self.environment = Environment(g, 0)
        self.agent = g.players[0]
        self.seed = seed
        self.stats = None
        self.exp_loss = []
        self.actual_regret = []
        self.exp_regret = []

    def run_interaction(self):
        strategy = self.agent.play_strategy()
        self.environment.observe_strategy(strategy)
        realization = self.agent.sample_strategy()
        self.environment.observe_realization(realization)
        feedback = self.environment.feedback("expert")
        self.agent.receive_feedback(feedback)

    def run(self):
        if self.game.is_finished():
            raise errors.FinishedGameError(self.game)
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
        df["exp_loss"] = self.exp_loss
        df["actual_regret"] = self.actual_regret
        df["exp_regret"] = self.exp_regret
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

    def compute_stats(self):
        self.exp_loss = exp_loss(self)
        self.actual_regret = actual_regret(self)
        self.exp_regret = exp_regret(self)
        self.stats = {}
        self.stats['total_rewards'] = self.total_rewards()
        self.stats['actual_regret'] = self.actual_regret[-1]
        self.stats['exp_regret'] = self.exp_regret[-1]

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed),
                        " stats:", str(self.stats), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed),
                        " stats:", str(self.stats), ">"])


class ResumedExperiment(Experiment):
    """
    An experiment re-built from the saved data

    Experiment can be resumed:
        - from an ended game
        - from a game and a result file
    The experiment is clearly complete in the first case, while in the second
    case only this features are available for now:
        - history
        - strategy_history
        - feedbacks 
        - seed: it is as it was at the beginning of the game!
    """

    def __init__(self, g, file=None, seed=None):
        if seed is None:
            if file is not None:
                seed = int(file.split("/")[-1])
        super().__init__(g, seed)
        if file is not None:
            df = pd.read_csv(file, index_col=0)
            for line in [r for i, r in df.iterrows()][1:]:
                strategy = dict()
                history = dict()
                strategy[0], history[0] = get_s_m(line[0])
                strategy[1], history[1] = get_s_m(line[1])
                self.game.history.append(history)
                self.game.strategy_history.append(strategy)
                T = len(self.game.values)
                feedbacks = {i: f for i, f in enumerate(line[2: 2 + T])}
                feedbacks['total'] = line[2 + T]
                self.agent.feedbacks.append(feedbacks)


def get_s_m(c):
    pattern = re.compile(r"^\((\[([0-9]|\.| |,)+\]), (\[[0-9]+\])\)$")
    pattern2 = re.compile("([0-9]+(\.[0-9]+)*)")
    s, m = pattern.match(c).group(1, 3)
    strategy = [float(t[0]) for t in pattern2.findall(s)]
    move = [int(t[0]) for t in pattern2.findall(m)]
    return strategy, move


class ResumedConfiguration(Configuration):

    def __init__(self, results_folder_path, print_results=False):
        self.game = game.load(results_folder_path + "/game")
        self.print_results = print_results
        self.experiments = []
        self.results_folder_path = results_folder_path
        self.stats = {}
        file_pattern = re.compile("[0-9]+")
        for f in os.listdir(self.results_folder_path + "/experiments"):
            file = self.results_folder_path +  + "/experiments/" + f
            if os.path.isfile(file) and bool(file_pattern.match(f)):
                g = deepcopy(self.game)
                self.experiments.append(ResumedExperiment(g, file))

    def del_prev_exp(self):
        shutil.rmtree(self.results_folder_path + "/experiments")
        os.makedirs(self.results_folder_path + "/experiments")
        self.experiments = []
        self.print_results = True


class ResumedBatch(Batch):
    def __init__(self, results_folder_path):
        self.print_results = False
        self.name = None
        self.parser = None
        self.configurations = []
        self.results_folder_path = results_folder_path
        for d in os.listdir(self.results_folder_path):
            directory = self.results_folder_path + "/" + d
            if os.path.isdir(directory):
                c = ResumedConfiguration(directory)
                self.configurations.append(c)
