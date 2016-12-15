import sys
from game import *
from player import *
from environment import *


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # Agent computes its strategy 
    strategy = agent.compute_strategy()
    environment.observe_strategy(strategy)
    realization = agent.sample_strategy()
    environment.observe_realization(realization)
    feedback = environment.feedback("expert")
    agent.receive_feedback(feedback)

if __name__ == "__main__":
    sys.exit(main())
