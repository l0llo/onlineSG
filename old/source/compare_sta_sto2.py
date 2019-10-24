import source.runner as runner

from source.util import *
import shutil
import os


def run(d, targets, time_horizon):
    # base_folder = "../games/sta_sto" + gen_name(d) + "/"
    base_folder = ("../games/" + print_targets(targets) + "/" +
                   gen_name(d).split("-", 1)[1])
    results_folder = base_folder  # + "results"
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    else:
        os.makedirs(base_folder)

    defenders = ["sta_ksto_mab1" + print_dist(d),
                  "sta_usto_mab",
                  "sta_ksto_expert1" + print_dist(d),
                  "sta_usto_expert",
                  "fabulous1" + print_dist(d),
                  "sta_sto_holmes11" + print_dist(d)]
    attackers = ["stackelberg", "stochastic_attacker1" + print_dist(d)]

    for a in attackers:
        batch_name = a
        batchpath = base_folder + "/" + batch_name + ".csv"
        with open(batchpath, "w+") as f:
            f.write("T," + gen_header(len(targets)) + "Defender,Attacker\n")
            for d in defenders:
                f.write(gen_game(targets, time_horizon) + d + "," + a + "\n")

    r = runner.Runner(base_folder, results_folder)
    r.run(100)

    for b in r.batches:
        comp = [c for c in b.configurations
                if isinstance(c, runner.Configuration)]
        plot_conf(exp_loss, comp, b.results_folder_path)
        plot_conf(actual_regret, comp, b.results_folder_path)
        plot_conf(exp_regret, comp, b.results_folder_path)
