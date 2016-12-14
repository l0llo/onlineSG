from random import uniform


class Player:

    def __init__(self, game, id, resources):
        """

        """
        self.game = game
        self.id = id
        self.resources = resources

    def compute_strategy(self):
        """
        set a probability distribution over the targets
        default: uniform strategy
        """
        targets_number = len(self.game.values)
        return [self.resources / targets_number for i in range(targets_number)]

    def sample_strategy(self):
        """
        sample a move from the computed distribution
        """
        targets = range(len(self.game.values))
        strategy = self.game.strategy_history[-1][self.id]
        sample = [uniform(0, strategy[i]) for i in targets]
        selected_targets = sorted(targets, key=lambda k: sample[k],
                                  reverse=True)[:self.resources]
        return selected_targets


class Attacker(Player):
    pass


class Defender(Player):
    pass
