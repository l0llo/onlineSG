class Environment:

    def __init__(self, game, agent_id):
        self.game = game
        self.agent_id = agent_id

    def observe_strategy(self, strategy):
        # Agent strategy is stored
        self.game.strategy_history.append(dict())
        self.game.strategy_history[-1][self.agent_id] = strategy

        # Attackers possibly observe and compute strategies
        for a in self.game.attackers:
            self.game.strategy_history[-1][a] = self.game.players[a].compute_strategy()

        # Players extract a sample from their strategies
        self.game.history.append(dict())
        for a in self.game.attackers:
            self.game.history[-1][a] = self.game.players[a].sample_strategy()

    def observe_realization(self, realization):
        self.game.history[-1][self.agent_id] = realization

    def feedback(self, feedback_type):
        targets = range(len(self.game.values))
        payoffs = self.game.get_last_turn_payoffs(self.agent_id)
        if feedback_type == "expert":
            return {t: payoffs[t] for t in targets}
        elif feedback_type == "partial":
            return {t: payoffs[t]
                    for t in self.game.history[-1][self.agent_id]}
