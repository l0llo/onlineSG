from source.runner import Experiment, Configuration, Batch
from copy import deepcopy
import os
import pandas as pd
import re
import source.game as game
import shutil


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
            self.exp_loss = list(df["exp_loss"])
            self.actual_regret = list(df["actual_regret"])
            self.exp_regret = list(df["exp_regret"])


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
            file = self.results_folder_path + "/experiments/" + f
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
