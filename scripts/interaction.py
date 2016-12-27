import sys
sys.path.append('../')

import source.game as game
import source.player as player
import source.environment as environment
import source.parsers as parsers
import source.runner as runner
import numpy as np
import pandas as pd
from importlib import *


def main(arguments):
    """
    SG with a Stackelberg attacker and a defender who play
    a uniform strategy
    """
    values = ((1, 1), (2, 2), (3, 3))
    time_horizon = 10
    g = game.Game(values, time_horizon)
    agent = player.Defender(g, 0, 1)
    attacker = player.StackelbergAttacker(g, 1, 1)
    g.set_players([agent], [attacker])
    e = environment.Environment(g, 0)
    time_horizon = 10

    for t in range(g.time_horizon):
        strategy = agent.compute_strategy()
        e.observe_strategy(strategy)
        realization = agent.sample_strategy()
        e.observe_realization(realization)
        feedback = e.feedback("expert")
        agent.receive_feedback(feedback)
    for i in g.history:
        print(i)


def main2(arguments):
    """
    SG with a Stackelberg attacker and a defender who can distinguish
    between a Stackelberg adn a Uniform
    """
    values = ((1, 1), (2, 2), (3, 3))
    time_horizon = 10
    g = game.Game(values, time_horizon)
    agent = player.StUDefender(g, 0)
    # attacker = player.StackelbergAttacker(g, 1)
    attacker = player.Attacker(g, 1)
    g.set_players([agent], [attacker])
    e = environment.Environment(g, 0)
    time_horizon = 10

    for t in range(g.time_horizon):
        strategy = agent.compute_strategy()
        e.observe_strategy(strategy)
        realization = agent.sample_strategy()
        e.observe_realization(realization)
        feedback = e.feedback("expert")
        agent.receive_feedback(feedback)
    print("history of the game")
    for i, h in enumerate(g.history):
        print("strategies at " + str(i) + ":")
        print("\t agent:", g.strategy_history[i][0], "\t attacker:", g.strategy_history[i][1])
        print("moves at :" + str(i) + ":")
        print("\t agent:", h[0], "\t attacker:", h[1])


if __name__ == '__main__':
    #main(sys.argv)
    main2(sys.argv)
