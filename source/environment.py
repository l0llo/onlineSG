import re
import sys
import socket


class Environment:

    def __init__(self, game, agent_id):
        self.game = game
        self.agent_id = agent_id
        self.last_exp_loss = None
        self.last_opt_loss = None
        self.last_act_loss = None

    def __str__(self):
        return ''.join(["<", self.__class__.__name__, ">"])

    def __repr__(self):
        return ''.join(["<", self.__class__.__name__, ">"])

    def observe_strategy(self, strategy):
        # Agent strategy is stored
        self.game.strategy_history.append(dict())
        self.game.strategy_history[-1][self.agent_id] = strategy

        # Attackers possibly observe and compute strategies
        for a in self.game.attackers:
            self.game.strategy_history[-1][a] = (self.game.players[a].
                                                 play_strategy())
        # hardcoded for 2 players
        str_dict = {0: strategy, 1: self.game.players[1].last_strategy}
        self.last_exp_loss = self.game.players[1].exp_loss(str_dict)
        self.last_opt_loss = self.game.players[1].opt_loss()
        # Profiles possibly observe and compute strategies
        for p in self.game.profiles:
            p.play_strategy()

        # Attackers extract a sample from their strategies
        self.game.history.append(dict())
        for a in self.game.attackers:
            self.game.history[-1][a] = self.game.players[a].sample_strategy()

    def observe_realization(self, realization):
        self.game.history[-1][self.agent_id] = realization
        for a in self.game.attackers:
            self.game.players[a].receive_feedback()
        for p in self.game.profiles:
            p.receive_feedback()

    def feedback(self, feedback_type):
        targets = range(len(self.game.values))
        payoffs = self.game.get_last_turn_payoffs(self.agent_id)
        self.last_act_loss = -sum(payoffs)
        if feedback_type == "expert":
            feedbacks = {t: payoffs[t] for t in targets}
            feedbacks['total'] = sum(feedbacks.values())
            return feedbacks
        elif feedback_type == "partial":
            feedbacks = {t: payoffs[t]
                         for t in self.game.history[-1][self.agent_id]}
            feedbacks['total'] = sum(feedbacks.values())
            return feedbacks
        elif feedback_type == "perceived":
            feedbacks = {t: payoffs[t] * self.game.observation_history[-1].get(t)
                         for t in targets}
            feedbacks['total'] = sum(feedbacks.values())
            return feedbacks

class RTEnvironment(Environment):

    def __init__(self, game, agent_id):
        super().__init__(game, agent_id)

    def observe_strategy(self, strategy):
        # Agent strategy is stored
        self.game.strategy_history.append(dict())
        self.game.strategy_history[-1][self.agent_id] = strategy

        # Attackers possibly observe and compute strategies
        cur_strategy = None
        while cur_strategy is None:
            print("insert a valid strategy")
            cur_strategy = self.parse_strategy(input())

        self.game.players[1].last_strategy = cur_strategy
        self.game.strategy_history[-1][1] = cur_strategy
        # Players extract a sample from their strategies
        self.game.history.append(dict())
        self.game.history[-1][1] = self.game.players[1].sample_strategy()
        # Profiles possibly observe and compute strategies
        for p in self.game.profiles:
            p.play_strategy()

    def parse_strategy(self, target_str):
        T = len(self.game.values)
        pattern = re.compile("^[0-" + str(T - 1) +"]$")
        if pattern.match(target_str):
            target = int(target_str)
            return [int(i == target) for i in range(T)]


class SocketEnvironment(RTEnvironment):

    def __init__(self, game, agent_id):
        super().__init__(game, agent_id)
        HOST = '192.168.0.12'   # Symbolic name meaning all available interfaces
        PORT = 8888  # Arbitrary non-privileged port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        try:
            self.socket .bind((HOST, PORT))
        except OSError as e:
            print('Bind failed. Error Code : ' + str(e.value[0]) +
                  ' Message ' + e.value[1])
            sys.exit()
        print('Socket bind complete')

        self.socket .listen(10)
        print('Socket now listening')

        #wait to accept a connection - blocking call
        self.conn, self.addr = self.socket.accept()

        #display client information
        print('Connected with ' + self.addr[0] + ':' + str(self.addr[1]))
        sent_data = (str([x[1] for x in self.game.values])
                     .split("[")[1].split("]")[0].encode("utf-8"))
        self.conn.sendall(sent_data)

    def observe_strategy(self, strategy):
        # Agent strategy is stored
        self.game.strategy_history.append(dict())
        self.game.strategy_history[-1][self.agent_id] = strategy

        sent_data = (str([round(x,2) for x in strategy])
                     .split("[")[1].split("]")[0].encode("utf-8"))
        self.conn.sendall(sent_data)
        rcv_data = self.conn.recv(1024).decode("utf-8")
        cur_strategy = self.parse_strategy(rcv_data)
        while cur_strategy is None:
            sent_data = "not valid".encode("utf-8")
            self.conn.sendall(sent_data)
            rcv_data = self.conn.recv(1024).decode("utf-8")
            cur_strategy = self.parse_strategy(rcv_data)
        self.game.players[1].last_strategy = cur_strategy
        self.game.strategy_history[-1][1] = cur_strategy
        # Players extract a sample from their strategies
        self.game.history.append(dict())
        self.game.history[-1][1] = self.game.players[1].sample_strategy()
        # Profiles possibly observe and compute strategies
        for p in self.game.profiles:
            p.play_strategy()

    def observe_realization(self, realization):
        #self.game.history[-1][self.agent_id] = realization
        super().observe_realization(realization)
        payoff = sum(self.game.get_last_turn_payoffs(1))
        sent_data = (str(realization[0]) + "," + str(payoff)).encode("utf-8")
        self.conn.sendall(sent_data)
