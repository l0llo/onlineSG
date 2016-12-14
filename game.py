from player import Player


class Game:
    """
    Attributes:

    values      tuple with a tuple for each target with the values for each player
    players     dict of players indexed by integers
    defenders   list of defenders' indexes
    attackers   list of attackers' indexes
    followers   list of followers' indexes
    history     list of dict for each turn: each one is made by the moves of the players
            (each move is a tuple of choosen targets indexes)

    For now we have a values tuple, but in general we could generalize and use a tensor with
    the players payoffs: this would mean changing also the get_player_payoff method
    """

    def __init__(self, payoffs, time_horizon):
        self.values = payoffs
        self.time_horizon = time_horizon
        self.players = dict()
        self.attackers = []
        self.followers = []
        self.defenders = []
        self.history = []
        self.strategy_history = []

    def play_turn(self):

        # Defenders compute strategies (it includes also computing rewards)
        self.strategy_history.append(dict())
        for d in self.defenders:
            self.strategy_history[-1][d] = self.players[d].compute_strategy()

        # Attackers possibly observe and compute strategies
        for a in self.attackers:
            self.strategy_history[-1][a] = self.players[a].compute_strategy()

        # Players extract a sample from their strategies
        self.history.append(dict())
        for p in self.players:
            self.history[-1][p] = self.players[p].sample_strategy()

    def play_game(self):
        for t in range(self.time_horizon):
            self.play_turn()

    def get_player_payoff(self, player_index, moves):
        """
        It returns the utility of a player given a dict
        of moves. Each move is a tuple of target indexes
        """
        covered_targets = set(t for d in self.defenders for t in moves[d])

        if player_index in self.attackers:
            hit_targets = set(t for t in moves[player_index]
                              if t not in covered_targets)
            return sum([v[player_index] for (i, v) in enumerate(self.values)
                        if i in hit_targets])
        elif player_index in self.defenders:
            all_hit_targets = set(t for a in self.attackers for t in moves[a]
                                  if t not in covered_targets)
            return sum([-(v[player_index]) for (i, v) in enumerate(self.values)
                        if i in all_hit_targets])
        else:
            raise Exception(
                "Cannot compute utility for an index than does not exit"
            )

    def get_last_turn_payoff(self, player_index):
        return self.get_player_payoff(player_index, self.history[-1])


def main():
    g = Game(((1, 1), (2, 2)), 3)
    d = Player(g, 0, 1)
    a = Player(g, 1, 1)
    g.players[0] = d
    g.players[1] = a
    g.defenders.append(0)
    g.attackers.append(1)
    g.play_game()
    print(g.history)


if __name__ == '__main__':
    main()
