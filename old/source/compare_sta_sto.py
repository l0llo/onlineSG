import source.runner as runner

import numpy as np
from importlib import *
from math import sqrt
from source.util import *
import shutil
import os
import matplotlib.pyplot as plt


def run(d):
    base_folder = "../games/sta_sto" + gen_name(d) + "/"
    results_folder = base_folder + "results"
    if os.path.exists(base_folder):
        shutil.rmtree()
    else:
        os.makedirs(base_folder)
    batch_name = "sta_sto"
    batchpath = base_folder + batch_name + ".csv"
    with open(batchpath, "w+") as f:
        f.write("T,0,1,2,Defender,Attacker\n")
        f.write("100,1,2,3,sta_ksto_mab1" + print_dist(d) + ",stackelberg\n")
        f.write("100,1,2,3,sta_usto_mab,stackelberg\n")
        f.write("100,1,2,3,sta_ksto_expert1" + print_dist(d) +
                ",stackelberg\n")
        f.write("100,1,2,3,sta_usto_expert,stackelberg\n")
        f.write("100,1,2,3,fabulous1" + print_dist(d) + ",stackelberg\n")

    batch_name = "sto_sta"
    batchpath = base_folder + batch_name + ".csv"
    with open(batchpath, "w+") as f:
        f.write("T,0,1,2,Defender,Attacker\n")
        f.write("100,1,2,3,sta_ksto_mab1" + print_dist(d) +
                ",stochastic_attacker1" + print_dist(d) + "\n")
        f.write("100,1,2,3,sta_usto_mab,stochastic_attacker1" +
                print_dist(d) + "\n")
        f.write("100,1,2,3,sta_ksto_expert1" + print_dist(d) +
                ",stochastic_attacker1" + print_dist(d) + "\n")
        f.write("100,1,2,3,sta_usto_expert,stochastic_attacker1" +
                print_dist(d) + "\n")
        f.write("100,1,2,3,fabulous1" + print_dist(d) +
                ",stochastic_attacker1" + print_dist(d) + "\n")

    r = runner.Runner(base_folder, results_folder)
    r.run(100)

    b = r.batches[0]
    comp = b.configurations
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    name = "payoffs"
    for i, c in enumerate(comp):
        payoffs = [np.array([stackelberg_loss(s[0], e.game)
                             for s in e.game.strategy_history])
                   for e in c.experiments]
        avg_payoffs = (sum(payoffs, np.zeros(len(payoffs[0]))) /
                       len(payoffs))
        variances = (sum([[(p - avg_payoffs[j]) * (p - avg_payoffs[j])
                           for j, p in enumerate(payoffs[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(payoffs[0]))) / (len(payoffs[0]) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(payoffs[0]))
                       for i, a in enumerate(avg_payoffs)]
        lower_bound = [a - z * sqrt(variances[i] / len(payoffs[0]))
                       for i, a in enumerate(avg_payoffs)]
        plt.plot(avg_payoffs, label=c.game.players[0].__class__.name,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name + ".png",
                bbox_inches='tight')
    plt.show()
    name = "regret"
    for i, c in enumerate(comp):
        regrets = [np.array(actual_regret(e, opt_loss_sta(e.game)))
                   for e in c.experiments]
        avg_regret = sum(regrets, np.zeros(len(regrets[0]))) / len(regrets)
        variances = (sum([[(p - avg_regret[j]) * (p - avg_regret[j])
                           for j, p in enumerate(regrets[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(regrets[0]))) /
                     (len(regrets[0]) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(regrets[0]))
                       for i, a in enumerate(avg_regret)]
        lower_bound = [a - z * sqrt(variances[i] / len(regrets[0]))
                       for i, a in enumerate(avg_regret)]
        plt.plot(avg_regret, label=c.game.players[0].__class__.name,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name +
                ".png", bbox_inches='tight')
    plt.show()
    name = "expected regret"
    for i, c in enumerate(comp):
        regrets = [np.array(stackelberg_regret(e, opt_loss_sta(e.game)))
                   for e in c.experiments]
        avg_regret = sum(regrets, np.zeros(len(regrets[0]))) / len(regrets)
        variances = (sum([[(p - avg_regret[j]) * (p - avg_regret[j])
                           for j, p in enumerate(regrets[i])] 
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(regrets[0]))) /
                     (len(regrets[0]) - 1))
        upper_bound = [a + z * sqrt(variances[i] / len(regrets[0]))
                       for i, a in enumerate(avg_regret)]
        lower_bound = [a - z * sqrt(variances[i] / len(regrets[0]))
                       for i, a in enumerate(avg_regret)]
        plt.plot(avg_regret, label=c.game.players[0].__class__.name,
                 color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound,
                         lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name + ".png",
                bbox_inches='tight')
    plt.show()

    b = r.batches[1]
    comp = b.configurations
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(comp))]
    z = 1.96
    name = "payoffs"
    for i, c in enumerate(comp):
        payoffs = [np.array([stochastic_loss(s, e.game)
                             for s in e.game.strategy_history])
                   for e in c.experiments]
        avg_payoffs = (sum(payoffs, np.zeros(len(payoffs[0]))) /
                       len(payoffs))
        variances = (sum([[(p - avg_payoffs[j]) * (p - avg_payoffs[j])
                           for j, p in enumerate(payoffs[i])]
                          for i, e in enumerate(c.experiments)],
                         np.zeros(len(payoffs[0]))) / (len(payoffs[0]) - 1))
        upper_bound = [a + z * sqrt(variances[i]/len(payoffs[0])) for i, a in enumerate(avg_payoffs)]
        lower_bound = [a - z * sqrt(variances[i]/len(payoffs[0])) for i, a in enumerate(avg_payoffs)]
        plt.plot(avg_payoffs, label=c.game.players[0].__class__.name, color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound, lower_bound, color=colors[i], alpha=0.3)
    #plt.plot([-1.2 for i in range(comp[0].experiments[0].game.time_horizon)], label="optimal")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name + ".png", bbox_inches='tight')
    plt.show()
    name = "regret"
    for i, c in enumerate(comp):
        regrets = [np.array(actual_regret(e, opt_loss_sto(e.game))) for e in c.experiments]
        avg_regret = sum(regrets, np.zeros(len(regrets[0]))) / len(regrets)
        variances = (sum([[(p-avg_regret[j])*(p-avg_regret[j])
                           for j,p in enumerate(regrets[i])] 
                          for i,e in enumerate(c.experiments)],np.zeros(len(regrets[0]))) / (len(regrets[0])-1))
        upper_bound = [a + z*sqrt(variances[i]/len(regrets[0])) for i,a in enumerate(avg_regret)]
        lower_bound = [a - z*sqrt(variances[i]/len(regrets[0])) for i,a in enumerate(avg_regret)]
        plt.plot(avg_regret, label = c.game.players[0].__class__.name, color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound, lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name + ".png", bbox_inches='tight')
    plt.show()
    name = "expected regret"
    for i,c in enumerate(comp):
        regrets = [np.array(stochastic_regret(e, opt_loss_sto(e.game))) for e in c.experiments]
        avg_regret = sum(regrets, np.zeros(len(regrets[0]))) / len(regrets)
        variances = (sum([[(p-avg_regret[j])*(p-avg_regret[j])
                           for j,p in enumerate(regrets[i])] 
                          for i,e in enumerate(c.experiments)],np.zeros(len(regrets[0]))) / (len(regrets[0])-1))
        upper_bound = [a + z*sqrt(variances[i]/len(regrets[0])) for i,a in enumerate(avg_regret)]
        lower_bound = [a - z*sqrt(variances[i]/len(regrets[0])) for i,a in enumerate(avg_regret)]
        plt.plot(avg_regret, label = c.game.players[0].__class__.name, color=colors[i])
        plt.fill_between(list(range(len(lower_bound))), upper_bound, lower_bound, color=colors[i], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel(name)
    plt.title(b.results_folder_path + "/" + name + ".png")
    plt.savefig(b.results_folder_path + "/" + name + ".png", bbox_inches='tight')
    plt.show()
