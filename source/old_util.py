import numpy as np
import os
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import pickle
from collections import namedtuple

"""
ALL WHAT HAVE BEEN DISCARDED FROM UTIL
"""


Difficulty = namedtuple("Difficulty", ["gap", "norm_gap", "similarity"])


def exp_loss(e):
    adv = e.game.players[1]
    return np.array([adv.exp_loss(s) for s
                     in e.game.strategy_history])


def actual_regret(e):
    regret = []
    adv = e.game.players[1]
    for f in e.agent.feedbacks:
        inst_regret = -f['total'] - adv.opt_loss()
        if len(regret) == 0:
            regret.append(inst_regret)
        else:
            regret.append(regret[-1] + inst_regret)
    return np.array(regret)


def exp_regret(e):
    regret = []
    adv = e.game.players[1]
    for s in e.game.strategy_history:
        inst_regret = adv.exp_loss(s) - adv.opt_loss()
        if len(regret) == 0:
            regret.append(inst_regret)
        else:
            regret.append(regret[-1] + inst_regret)
    return np.array(regret)


def avg(e):
    return [sum([f['total'] for f in e.agent.feedbacks[:i]]) / i
            for i in list(range(1, len(e.game.history)))]


def gen_header(l):
    return "".join(str(i) + "," for i in range(l))


def plot_conf(fun, comp, path, name=None):
    plt.close()
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    if name is None:
        name = fun.__name__
    for i, c in enumerate(comp):
        f = [fun(e) for e in c.experiments]
        avg_f = (sum(f, np.zeros(len(f[0]))) / len(f))
        variances = (sum([[(p - avg_f[j]) * (p - avg_f[j])
                           for j, p in enumerate(f[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(f[0]))) / (len(f) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        lower_bound = [a - z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        plt.plot(avg_f, label=c.name,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(fun.__name__)
    plt.title(path + "/" + name + ".png" + "\n" +
              str([v[0] for v in comp[0].game.values]))
    plt.savefig(path + "/" + name + ".png",
                bbox_inches='tight')
    plt.show()

def plot_conf2(fun_str, comp, path, name=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    if name is None:
        name = fun_str
    for i, c in enumerate(comp):
        f = [e.__dict__[fun_str] for e in c.experiments]
        avg_f = (sum(f, np.zeros(len(f[0]))) / len(f))
        variances = (sum([[(p - avg_f[j]) * (p - avg_f[j])
                           for j, p in enumerate(f[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(f[0]))) / (len(f) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        lower_bound = [a - z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        ax.plot(avg_f, label=c.name,
                color=colors[i])
        ax.fill_between(list(range(len(lower_bound))), upper_bound,
                        lower_bound, color=colors[i], alpha=0.3, label=c.name)
    ax.set_ylabel(fun_str)
    ax.set_title(path + "/" + name + ".png" + "\n")
    handles, labels = ax.get_legend_handles_labels()
    handles = [h for h in handles if isinstance(h,
                                                matplotlib.lines.Line2D)]
    labels = [h._label for h in handles]
    ax.legend(loc=2, bbox_to_anchor=(1, 1), borderaxespad=0.,
              fancybox=True, shadow=True, handles=handles, labels=labels)
    fig.savefig(path + "/" + name + ".png",
                bbox_inches='tight')
    with open(path + "/plot", mode='w+b') as file:
        pickle.dump(fig, file)
    plt.close(fig)


def plot_conf(fun_str, comp, path, name=None, show=False):
    plt.close()
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    if name is None:
        name = fun_str
    for i, c in enumerate(comp):
        f = [e.__dict__[fun_str] for e in c.experiments]
        avg_f = (sum(f, np.zeros(len(f[0]))) / len(f))
        variances = (sum([[(p - avg_f[j]) * (p - avg_f[j])
                           for j, p in enumerate(f[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(f[0]))) / (len(f) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        lower_bound = [a - z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        plt.plot(avg_f, label=c.name,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(fun_str)
    plt.title(path + "/" + name + ".png" + "\n")  # +
              #str([v[0] for v in comp[0].game.values]))
    plt.savefig(path + "/" + name + ".png",
                bbox_inches='tight')
    if show:
        plt.show()


def plot_from_csv(fun, comp, path, name=None):
    plt.close()
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    if name is None:
        name = fun.__name__
    for i, c in enumerate(comp):
        folder = c + "/experiments"
        experiments_dfs = [pd.read_csv(folder + "/" + e)
                           for e in os.listdir(folder)
                           if e != "seeds.txt"]
        f = np.array([df[fun.__name__] for df in experiments_dfs])
        avg_f = (sum(f, np.zeros(len(f[0]))) / len(f))
        variances = (sum([[(p - avg_f[j]) * (p - avg_f[j])
                           for j, p in enumerate(f[i])]
                          for i, e in enumerate(experiments_dfs)],
                         np.zeros(len(f[0]))) / (len(f) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        lower_bound = [a - z * sqrt(variances[i] / len(f))
                       for i, a in enumerate(avg_f)]
        plt.plot(avg_f, label=c,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,)
    plt.ylabel(fun.__name__)
    plt.title(path + "/" + name + ".png" + "\n" +
              str([v[0] for v in comp[0].game.values]))
    plt.savefig(path + "/" + name + ".png",
                bbox_inches='tight')
    plt.show()


def print_move(g, m=None):
    if m is None:
        m = g.history[-1]
    l1 = ""
    for i in range(len(g.values)):
        if m[0][0] == i:
            l1 += "□\t"
        else:
            l1 += " \t"
    l2 = "".join([str(i[0]) + "\t" for i in g.values])
    l3 = ""
    for i in range(len(g.values)):
        if m[1][0] == i:
            l3 += "△\t"
        else:
            l3 += " \t"
    print(l1)
    print(l2)
    print(l3)

def cos_sim(d1, d2):
    return (np.array(d1).dot(np.array(d2)) /
            ((np.linalg.norm(np.array(d1))) *
             (np.linalg.norm(np.array(d2)))))


def difficulty(a1, a2):
    import source.players.attackers as attackers
    #gap1 = a1.opt_loss() - a2.opt_loss()
    def2_str = a2.get_best_responder().compute_strategy()
    gap2 = a1.exp_loss({0: def2_str, 1: None}) - a1.opt_loss()
    norm = max([v[0] for v in a1.game.values]) - a1.opt_loss()
    norm_gap = gap2 / norm
    similarity = (cos_sim(a1.distribution, a2.distribution)
                  if (a1.__class__.name == a2.__class__.name ==
                      attackers.StochasticAttacker.name) else 0)
    return Difficulty(gap2, norm_gap, similarity)


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def translate(targets):
    import source.game as game

    m = min(targets)
    r = round(float(np.random.uniform(high=m)), 3)
    targets2 = [x - r for x in targets]
    values = tuple((v, v) for v in targets2)
    g = game.Game(values, 1)
    g.attackers = [1]
    g.defenders = [0]
    if len(targets) == len(support(g)):
        return targets2
    else:
        return translate(targets)


def gen_setting(T, P, targets_dict, time_horizon):

    import source.players.attackers as attackers
    import source.game as game

    targets = translate(targets_dict[T][0])
    distributions = []
    for i in range(P):
        distributions.append(tuple(gen_distr(T)))
    values = tuple((v, v) for v in targets)
    g = game.Game(values, time_horizon)
    g.attackers = [1]
    g.defenders = [0]
    #print(T, targets)
    att = [attackers.StackelbergAttacker(g, 1)]
    for d in distributions:
        #print(d)
        att.append(attackers.StochasticAttacker(g, 1, 1, *d))
    profiles = []
    index = 0  # hardcoded for stackelberg
    profiles.append({"attacker": print_adv(att[index]),
                     "others": [print_adv(a) for a in att
                                if a != att[index]], "i": index})
    return targets, profiles