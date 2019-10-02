import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import pickle
from math import sqrt
from collections import namedtuple
import random
import collections
from functools import reduce
import operator

"""
A collection of useful function
"""

# PLOT

matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('text', usetex=True)

t = {'br_expert1-1': 'FPL',
     'br_mab1-1': 'UCB1',
     'holmes1-1': 'FR',
     'mbb2bw2w1': 'BBF',
     'mfb2bw2w1': 'FB'}


def plot_dicts(dlst, name="figure", ylabel="$R(U)_n$", path=".",
               alpha=0.3, cm='gnuplot', xlabel="$n$", t=None,
               save=False, show=True, semilog=True, title=""):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.85])
    plt.title(title)
    if semilog:
        ax.set_yscale('log')
    cmap = plt.get_cmap(cm)
    colors = [cmap(i) for i in np.linspace(0, 0.6, len(dlst))]
    markers = ['^', 'x', 'd', '*', 'h', 'o']
    min_avg = 0.9 * min([dl['avgs'][2] for dl in dlst])
    for i, d in enumerate(dlst):
        length = len(d["avgs"])
        markevery = list(range(0, length, int(length / 10)))
        ax_label = t[d["name"]] if t else d["name"]
        ax.plot(list(range(len(d["avgs"])))[2:], d["avgs"][2:],
                linestyle=':', label=ax_label,
                color=colors[i], marker=markers[i],
                markevery=markevery)
        if ("tb" in d) and (d["tb"] is not None):
            ax.plot(list(range(len(d["tb"])))[2:], d["tb"][2:],
                    linestyle="solid",
                    color=colors[i])
        # ax.fill_between(list(range(len(d["lb"]))), d["ub"],
        #                d["lb"], color=colors[i], alpha=0.3, label=d["name"])
        if ("lb" in d) and ("ub" in d):
            ax.fill_between(list(range(len(d["lb"])))[2:], d["ub"][2:],
                            [max(lb, min_avg) for lb in d["lb"]][2:],
                            color=colors[i], alpha=alpha, label=d["name"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
#    ax.set_ylim([0, 1])
    handles, labels = ax.get_legend_handles_labels()
    handles = [h for h in handles if isinstance(h,
                                                matplotlib.lines.Line2D)]
#    labels = [h._label for h in handles]
#    labels = ["$Full$", "$MAB$"]
    labels = ["$Sto$", "$Fusto$", "$Busto$", "$Busto\_good$"]
    ax.legend(loc=2, bbox_to_anchor=(0, 1), borderaxespad=0.1,
              fancybox=False, shadow=False, handles=handles,
              labels=labels, prop={'size': 9})
    if save:
        fig.savefig(path + "/" + name + ".pdf",
                    bbox_inches='tight', format="pdf")

        with open(path + "/plot", mode='w+b') as file:
            pickle.dump(fig, file)
    if show:
        plt.show(fig)
    plt.close(fig)


def avg_with_conf(lst, name=""):
    avgs = sum(lst, np.zeros(len(lst[0]))) / len(lst)
    variances = [np.array([(r - avgs[i]) ** 2 for i, r in enumerate(ls)])
                 for ls in lst]
    avg_var = sum(variances) / max((len(variances) - 1), 1)
    z = 1.96
    upper_bound = [a + z * sqrt(avg_var[i] / len(variances))
                   for i, a in enumerate(avgs)]
    lower_bound = [max(a - z * sqrt(avg_var[i] / len(variances)), 0)
                   for i, a in enumerate(avgs)]
    return {"avgs": avgs, "ub": upper_bound, "lb": lower_bound, "name": name}


# GAME GENERATION


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
                              best_response())]
    return [g.values[i][0] for i, t in enumerate(strategies) if t]


def gen_distr(n):
    d = (np.random.dirichlet([1 for i in range(n)]))
    d = np.array([round(x, 3) for x in d])

    d /= np.linalg.norm(d, ord=1)
    return list(d)


def gen_targets(l, low=0, high=100):
    targets = []
    for i in range(l):
        t = np.random.randint(low, high)
    # no targets with the same value (for now)
        while t in targets:
            t = np.random.randint(low, high)
        targets.append(t)
    return targets


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

def gen_probabilities_with_len(length):
    """
    Just a simple function to randomly generate distinct observability probabilities
    """
    if length <= 0:
        print("Length must be positive!")
    else:
        probabilities = dict()
        for i in range(length):
            p = round(np.random.uniform(0.7, 1), 3) #edit first parameter of np.random.uniform to set minimum threshold
            while p in probabilities:
                p = round(np.random.uniform(0.7, 1), 3)
            probabilities[i] = p
        return probabilities

def sample_probability(probabilities):
    sample = dict()
    for t in range(len(probabilities)):
        sample[t] = np.random.choice(2, p=[1 - probabilities.get(t), probabilities.get(t)])
    return sample

def gen_observabilities_correlated_with_values(values):
    if not isinstance(values, tuple):
        print("Inserted values must be a tuple")
    else:
        observabilities = dict()
        att_values =[v[1] for v in values]
        max_v = max(att_values)
        min_v = min(att_values)
        quarter = (max_v - min_v) / 4
        for i in range(len(values)):
            if att_values[i] >= min_v + 3 * quarter:
                o = round(np.random.uniform(0.75, 1), 3)
            elif att_values[i] >= min_v + 2 * quarter:
                o = round(np.random.uniform(0.5, 0.75), 3)
            elif att_values[i] >= min_v + quarter:
                o = round(np.random.uniform(0.25, 0.5), 3)
            else:
                o = round(np.random.uniform(0, 0.25), 3)
            observabilities[i] = o
        return observabilities


def gen_pdict(g, prof_list):
    import source.players.attackers as atk
    import source.parsers as parsers
    Prof = namedtuple("Prof", ["prof", "adv"])
    atk_dict = dict()
    for p in prof_list:
        if isinstance(p, tuple):
            profile = parsers.parse_player(p[0], g, 1)
            attacker = parsers.parse_player(p[1], g, 1)
            atk_dict[p[0]] = Prof(prof=profile, adv=attacker)
        else:
            profile = parsers.parse_player(p, g, 1)
            atk_dict[p] = Prof(prof=profile, adv=profile.get_attacker())
    return atk_dict


# def gen_pdict(g, prof_list):
#     import source.players.attackers as atk
#     Prof = namedtuple("Prof", ["prof", "adv"])
#     suqr_atk = atk.SUQR(g, 1)
#     sto_atk = atk.StochasticAttacker(g, 1)
#     atk_dict = {
#                 "usto": Prof(prof=atk.UnknownStochasticAttacker(g, 1),
#                              adv=atk.StochasticAttacker(g, 1)),
#                 "sta": Prof(prof=atk.StackelbergAttacker(g, 1),
#                             adv=atk.StackelbergAttacker(g, 1)),
#                 "suqr": Prof(prof=suqr_atk, adv=suqr_atk),
#                 "sto": Prof(prof=sto_atk, adv=sto_atk),
#                 "fp": Prof(prof=atk.FictitiousPlayerAttacker(g, 1),
#                            adv=atk.FictitiousPlayerAttacker(g, 1)),
#                 "usuqr": Prof(prof=atk.USUQR(g, 1), adv=atk.SUQR(g, 1))
#                 }
#     return {p: atk_dict[p] for p in prof_list}


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
            profiles.append(c(mock_game, 1))
    return profiles

# CONFIG FILE GENERATION


def gen_name(distribution):
    return ''.join(["-" + str(d).split('.')[1] for d in distribution])


def print_targets(targets):
    return str(targets[0]) + "".join(["_" + str(t) for t in targets[1:]])


def print_adv(adv):

    import source.players.attackers as attackers

    if adv.__class__.name == attackers.StochasticAttacker.name:
        return (adv.__class__.name + str(adv.resources) +
                print_dist(adv.distribution))
    elif adv.__class__.name == attackers.SUQR.name:
        return (adv.__class__.name + str(adv.resources) +
                print_dist((adv.w1, adv.w2)))
    else:
        return adv.__class__.name + str(adv.resources)


def print_row(targets, time_horizon, d, p):
    return (",".join([str(i) for i in ([d + "_vs_" + str(p["i"])] +
                                       [time_horizon] + targets + [d] +
                                       [p["attacker"]] +
                                       [x for x in p["others"]])]) + "\n")


def gen_conf_row(name, time_horizon, targets, def_str, att_str,
                 pro_str_lst, att_incl=False):
    if att_incl:
        pro_str_lst.append(att_str)
    return ",".join([name, str(time_horizon)] + [str(t) for t in targets] +
                    [def_str, att_str] + pro_str_lst) + "\n"


def print_header(targets, P, att_incl=False):
    return ("Name,T," + ",".join(str(i) for i in range(len(targets))) +
            ",Defender,Attacker," +
            ",".join(["Profile" for x in range(P + int(att_incl))]) +
            "\n")


def print_dist(distribution):
    return''.join(["-" + str(d) for d in distribution])


# PRETTY PRINT AND MISCELLANEOUS

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


def end_sound():
    os.system("( speaker-test -t sine -f 1000 )& pid=$! ;" +
              " sleep 0.2s ; kill -9 $pid")


def get_el(d, lst):
    # get an element from a multidimensional dict
    if not lst:
        return d
    else:
        return get_el(d[lst[0]], lst[1:])


def norm_min(v, m=0.01):
    v = np.array(v)
    n = np.linalg.norm(v, ord=1)
    v = v / n
    gr = []
    n1 = 0
    for i, x in enumerate(v):
        if x >= m:
            n1 += x
            gr.append(i)
    c = (1 - (len(v) - len(gr)) * m) / n1
    v1 = [x * c if i in gr else m for i, x in enumerate(v)]
    return v1



def two_largest_diff(numbers):
    count = 0
    max1 = max2 = float('-inf')
    for x in numbers:
        count += 1
        if x > max2:
            if x >= max1:
                max1, max2 = x, max1
            else:
                max2 = x
    return abs(max1 - max2) if count >= 2 else float('+inf')


def rand_max(iterable, key=None):
    """
    Works the same as max, but if you pass dictionary keys and finds multiple
    maxima among the values, it randomly returns one of the keys instead of just
    the first one
    """
    if len(iterable) == 1:
        return(list(iterable)[0])
    it = iter(iterable)
    try:
        max_val = next(it)
    except StopIteration:
        raise ValueError("rand_max() called with empty iterable")
    if key is None:
        for val in it:
            if val > max_val:
                max_val = val
    else:
        max_keyval = key(max_val)
        max_val = [max_val]
        for val in it:
            keyval = key(val)
            if keyval > max_keyval:
                max_val = [val]
                max_keyval = keyval
            elif keyval == max_keyval:
                max_val.append(val)
        max_val = random.choice(max_val)
    return max_val

def find_min_diff(arr, n):
    arr = sorted(arr)
    diff = 10 ** 20
    for i in range(n - 1):
        if arr[i + 1] - arr[i] < diff:
            diff = arr[i + 1] - arr[i]
    return diff

def flatten(l):
    """
    Flattens nested list independently on the nesting level;
    returns a generator of the nested list
    """
    for el in l:
        if (isinstance(el, collections.Iterable)
            and not isinstance(el, (str, bytes))):
            yield from flatten(el)
        else:
            yield el

def prod(iterable):
    """
    Analogous to sum(iterable), but returns the product of the elements instead
    """
    return reduce(operator.mul, iterable, 1)
