import numpy as np
import os
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
import pickle
from collections import namedtuple
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


# def avg(e):
#     return [sum([f['total'] for f in e.agent.feedbacks[:i]]) / i
#             for i in list(range(1, len(e.game.history)))]


def print_dist(distribution):
    return''.join(["-" + str(d) for d in distribution])


def gen_distr(n):
    d = (np.random.dirichlet([1 for i in range(n)]))
    d = np.array([round(x, 3) for x in d])
    d /= np.linalg.norm(d, ord=1)
    return list(d)


def gen_name(distribution):
    return ''.join(["-" + str(d).split('.')[1] for d in distribution])


def print_targets(targets):
    return str(targets[0]) + "".join(["_" + str(t) for t in targets[1:]])


def gen_targets(l, low=0, high=100):
    targets = []
    for i in range(l):
        t = np.random.randint(low, high)
    # no targets with the same value (for now)
        while t in targets:
            t = np.random.randint(low, high)
        targets.append(t)
    return targets


def gen_header(l):
    return "".join(str(i) + "," for i in range(l))


# def plot_conf(fun, comp, path, name=None):
#     plt.close()
#     cmap = plt.get_cmap('gnuplot')
#     colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
#     z = 1.96
#     if name is None:
#         name = fun.__name__
#     for i, c in enumerate(comp):
#         f = [fun(e) for e in c.experiments]
#         avg_f = (sum(f, np.zeros(len(f[0]))) / len(f))
#         variances = (sum([[(p - avg_f[j]) * (p - avg_f[j])
#                            for j, p in enumerate(f[i])]
#                           for i, e in enumerate(c.experiments)],
#                          np.zeros(len(f[0]))) / (len(f) - 1))
#         upper_bound = [a + z * sqrt(variances[i] / len(f))
#                        for i, a in enumerate(avg_f)]
#         lower_bound = [a - z * sqrt(variances[i] / len(f))
#                        for i, a in enumerate(avg_f)]
#         plt.plot(avg_f, label=c.name,
#                  color=colors[i])
#         plt.fill_between(list(range(len(lower_bound))), upper_bound,
#                          lower_bound, color=colors[i], alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.ylabel(fun.__name__)
#     plt.title(path + "/" + name + ".png" + "\n" +
#               str([v[0] for v in comp[0].game.values]))
#     plt.savefig(path + "/" + name + ".png",
#                 bbox_inches='tight')
#     plt.show()

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


def move_str(g, m=None):
    if m is None:
        m = g.history[-1]
    l1 = ""
    for i in range(len(g.values)):
        if m[0][0] == i:
            l1 += "□\t"
        else:
            l1 += " \t"
    l2 = "".join([str(round(i[0], 3)) + "\t" for i in g.values])
    l3 = ""
    for i in range(len(g.values)):
        if m[1][0] == i:
            l3 += "△\t"
        else:
            l3 += " \t"
    return "\n".join([l1, l2, l3])


def game_str(g, start=None, end=None, lenght=8):
    sep = "\n" + "-" * len(g.values) * lenght + "\n"
    return sep.join([move_str(g, h) for h in g.history[start:end]])


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


def print_adv(adv):

    import source.players.attackers as attackers

    if adv.__class__.name == attackers.StochasticAttacker.name:
        return (adv.__class__.name + str(adv.resources) +
                print_dist(adv.distribution))
    else:
        return adv.__class__.name + str(adv.resources)


def end_sound():
    os.system("( speaker-test -t sine -f 1000 )& pid=$! ; sleep 0.2s ; kill -9 $pid")


def gen_norm_targets(l):
    targets = []
    for i in range(l):
        t = np.random.random()
    # no targets with the same value (for now)
        targets.append(t)
    return targets


def support(g):
    import source.players.attackers as attackers

    strategies = [t for t in (attackers.StackelbergAttacker(g, 1).
                              get_best_responder().compute_strategy())]
    return [g.values[i][0] for i, t in enumerate(strategies) if t]


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


def print_row(targets, time_horizon, d, p):
    return (",".join([str(i) for i in ([d +"_vs_" + str(p["i"])] +
                                       [time_horizon] + targets + [d] +
                                       [p["attacker"]] +
                                       [x for x in p["others"]])]) + "\n")


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


def gen_conf_row(name, time_horizon, targets, def_str, att_str,
                 pro_str_lst, att_incl=False):
    if att_incl:
        pro_str_lst.append(att_str)
    return ",".join([name, str(time_horizon)] + [str(t) for t in targets] +
                    [def_str, att_str] + pro_str_lst) + "\n"


def print_header(targets, P, att_incl=False):
    return ("Name,T," + ",".join(str(i) for i in range(len(targets))) +
            ",Defender,Attacker," +
            ",".join(["Profile" for x in range(P + int( att_incl))]) +
            "\n")


def gen_profiles(targets, p_pair_lst):
    """
    the p_pair_lst is formed by pairs of this type:
    (attacker_class, number)
    which indicate what should be the composition of the profiles list
    """
    import source.game as game

    mock_game = game.zs_game(targets, 1)
    profiles = []
    for c, n in p_pair_lst:
        for i in range(n):
            profiles.append(c(mock_game, 1, 1))
    return profiles


def gen_tar_with_len(length):
    import source.game as game

    len_s = 0
    while len_s != length:
        T = 15
        if length == 10:
            T = 50
        time_horizon = 10
        targets = [round(x, 3) for x in gen_norm_targets(T)]
        values = tuple((v, v) for v in targets)
        g = game.Game(values, time_horizon)
        g.attackers = [1]
        g.defenders = [0]
        s = support(g)
        len_s = len(s)
    return s
