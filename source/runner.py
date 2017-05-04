from source.environment import *
from collections import namedtuple
from os import listdir
from os.path import isfile, join
from source.errors import UnknownHeaderError, RowError, FolderExistsError
from copy import deepcopy, copy
from shutil import copyfile
from math import sqrt
import source.parsers as parsers
import os
import pandas as pd
import traceback
import numpy as np
# import shutil
import concurrent.futures
import logging
import time


AbortedExperiment = namedtuple('AbortedExperiment', ['error', 'info', 'seed'])
AbortedConfiguration = namedtuple('AbortedConfiguration', ['error', 'info', 'row'])
AbortedBatch = namedtuple('AbortedBatch', ['error', 'file'])

logger = logging.getLogger(__name__)


class Runner:

    def __init__(self, runnables=[], name=""):
        # list of Runnables, can be Batches or other Runners
        self.runnables = runnables
        self.name = name

    def run(self, workers=None, sub_args_dict={}):
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            futures = []
            for r in self.runnables:
                futures.append(executor.submit(r.__class__.run,
                                               r, **sub_args_dict))
        concurrent.futures.wait(futures)

        # If it is necessary to do something with them, use this instead
        # for future in concurrent.futures.as_completed(futures):
        #     result = future.result()


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

    # def run(self, n=1, show_progress=False, workers=None):
    #     for c in self.configurations:
    #         if isinstance(c, Configuration):
    #             c.run(n, workers)

    def run(self, futures, executor, n=1):
        for c in self.configurations:
            if isinstance(c, Configuration):
                c.run(futures, executor, n)

    def collect(self, futures):
        for c in self.configurations:
            c.collect(futures)

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

    def __init__(self, g, folder=None, name=None, print_results=True):
        self.print_results = print_results
        self.game = g
        self.experiments = []
        self.name = name
        if folder is not None:
            self.results_folder_path = folder + "/" + name
        if print_results:
            if os.path.exists(self.results_folder_path + "/experiments"):
                raise FolderExistsError(self.results_folder_path)
            else:
                os.makedirs(self.results_folder_path + "/experiments")
            pickle_file = self.results_folder_path + "/game"
            self.game.dump(pickle_file)

            # json_file = self.results_folder_path + "/json.txt"
            # self.game.dumpjson(json_file)  # never used in practice
        self.stats = {}

    def compute_stats(self):

        # self.stats['avg_run_time'] = sum([e.run_time
        #                                   for e in self.experiments
        #                                   if isinstance(e, Experiment)])
        # self.stats['avg_total_rewards'] = sum([e.total_rewards()
        #                                        for e in self.experiments
        #                                        if isinstance(e, Experiment)])
        # self.stats['avg_exp_regret'] = sum([e.exp_regret[-1]
        #                                     for e in self.experiments
        #                                     if isinstance(e, Experiment)])

        self.stats["exp_loss"] = (sum([np.array(e.exp_loss)
                                       for e in self.experiments
                                       if isinstance(e, Experiment)]) /
                                  len(self.experiments))
        self.stats["actual_regret"] = (sum([np.array(e.actual_regret)
                                            for e in self.experiments
                                            if isinstance(e, Experiment)]) /
                                       len(self.experiments))
        lst = [np.array(e.exp_regret) for e in self.experiments
               if isinstance(e, Experiment)]
        avgs = sum(lst) / len(lst)
        var = [np.array([(r - avgs[i]) ** 2
                         for i, r in enumerate(ls)])
               for ls in lst]
        avg_var = (sum(var) / (len(var) - 1))
        z = 1.96
        upper_bound = [a + z * sqrt(avg_var[i] / len(var))
                       for i, a in enumerate(avgs)]
        lower_bound = [max(a - z * sqrt(avg_var[i] / len(var)), 0)
                       for i, a in enumerate(avgs)]

        self.stats["exp_regret"] = avgs
        self.stats["lb_exp_regret"] = lower_bound
        self.stats["ub_exp_regret"] = upper_bound

        # exp_number = len([e for e in self.experiments
        #                   if isinstance(e, Experiment)])
        # if exp_number:
        #     self.stats['avg_total_rewards'] /= exp_number
        #     self.stats['avg_exp_regret'] /= exp_number
        #     self.stats['avg_run_time'] /= exp_number

    def run_an_experiment(self, seed=None):
        if not seed:
            seed = init_seed()
        g = deepcopy(self.game)
        experiment = Experiment(g, seed=seed)
        try:
            experiment.run()
            if self.print_results:
                experiment.save_results(self.results_folder_path + "/experiments")
        except Exception as e:
            print("Something has gone wrong with the experiment: ", e)
            return AbortedExperiment(e, traceback.format_exc(), seed)
        else:
            return experiment

    # def run(self, n=1, workers=None):
    #     with concurrent.futures.ProcessPoolExecutor(workers) as executor:
    #         futures = []
    #         for i in range(n):
    #             futures.append(executor.submit(Configuration.run_an_experiment,
    #                                            self))
    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #         self.experiments.append(result)
    #     self.compute_stats()
    #     if self.print_results:
    #         self.print_stats_file()

    def run(self, futures, executor, n=1):
        futures[self] = []
        for i in range(n):
            futures[self].append(executor.submit(Configuration
                                                 .run_an_experiment,
                                                 self))

    def collect(self, futures):
        for future in concurrent.futures.as_completed(futures[self]):
            result = future.result()
            self.experiments.append(result)
        self.compute_stats()
        if self.print_results:
            self.print_stats_file()

    def print_stats_file(self):
        df = pd.DataFrame()
        df["exp_loss"] = self.stats["exp_loss"]
        df["actual_regret"] = self.stats["actual_regret"]
        df["exp_regret"] = self.stats["exp_regret"]
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

    def __init__(self, g, env=None, seed=None):
        if not seed:
            seed = init_seed()
        else:
            np.random.seed(seed)
        self.game = g
        if env is None:
            self.environment = Environment(g, 0)
        elif env == "rt":
            self.environment = RTEnvironment(g, 0)
        elif env == "socket":
            self.environment = SocketEnvironment(g, 0)
        self.agent = g.players[0]
        self.seed = seed
        #self.stats = None
        self.exp_loss = []
        self.actual_regret = []
        self.exp_regret = []
        self.run_time = 0

    def run_interaction(self):
        strategy = self.agent.play_strategy()
        self.environment.observe_strategy(strategy)
        realization = self.agent.sample_strategy()
        self.environment.observe_realization(realization)
        feedback = self.environment.feedback("expert")
        self.agent.receive_feedback(feedback)
        self.update_stats()

    def update_stats(self):
        el = self.environment.last_exp_loss
        ol = self.environment.last_opt_loss
        al = self.environment.last_act_loss
        if (el, ol, al) != (None, None, None):
            self.exp_loss.append(el)
            if not self.exp_regret:
                self.exp_regret.append(el - ol)
            else:
                self.exp_regret.append(self.exp_regret[-1] + (el - ol))
            if not self.actual_regret:
                self.actual_regret.append(al - ol)
            else:
                self.actual_regret.append(self.actual_regret[-1] + (al - ol))

    def run(self, verbose=True):
        global logger

        start_time = time.time()
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
        if self.game.is_finished():
            raise errors.FinishedGameError(self.game)
        i = 0
        logger.info("T:" + str(len(self.game.values)) +
                    " P:" + str(len(self.game.profiles)) +
                    self.agent.__class__.name)
        while(not self.game.is_finished()):
            i += 1
            # if not i % 100:
            #     logger.info("round: " + str(i))
            self.run_interaction()
        #self.compute_stats()
        self.run_time = time.time() - start_time

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
        return (pd.read_csv(folder + "/" + str(self.seed), index_col=0).
                iloc[start:end])

    def total_rewards(self):
        return sum([f['total'] for f in self.agent.feedbacks])

    def __str__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed), ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__,
                        " seed:", str(self.seed), ">"])
