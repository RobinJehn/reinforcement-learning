import random
from typing import Dict, Set

import numpy as np

from tictactoe import Player, TicTacToe


class Agent:
    def __init__(self, exploration: float = 0.5, exploration_decay: float = 0.99):
        self.value_function = ValueFunction()
        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.previous_state: Dict[Player, TicTacToe] = {
            Player.X: TicTacToe(),
            Player.O: TicTacToe(),
        }
        self.alpha = 0.1

    def get_move(self, game: TicTacToe, player: Player):
        valid_moves = game.get_valid_moves()

        best_move = None
        best_value = -1 if player == Player.X else 1
        for valid_move in valid_moves:
            hypothetical_game = game.copy()
            hypothetical_game.make_move(valid_move[0], valid_move[1])
            value = self.value_function.get_value(hypothetical_game)
            if player == Player.X:
                if value > best_value:
                    best_value = value
                    best_move = valid_move
            else:
                if value < best_value:
                    best_value = value
                    best_move = valid_move

        chosen_move = None
        if random.random() < self.exploration:
            move_index = random.randrange(0, len(valid_moves))
            chosen_move = valid_moves[move_index]
        else:
            chosen_move = best_move

        self.exploration *= self.exploration_decay

        new_game = game.copy()
        new_game.make_move(chosen_move[0], chosen_move[1])
        self.value_function.update_value(
            self.previous_state[player], new_game, self.alpha
        )
        self.previous_state[player] = new_game

        return chosen_move

    def lost_game(self, game: TicTacToe, player: Player):
        self.value_function.update_value(self.previous_state[player], game, self.alpha)


class ValueFunction:
    def __init__(self):
        open_states: Set[TicTacToe] = set()
        open_states.add(TicTacToe())

        self.value: Dict[TicTacToe, float] = {}

        while len(open_states) > 0:
            state = open_states.pop()

            if state in self.value:
                continue

            match state.get_winner():
                case None:
                    self.value[state] = 0
                case Player.X:
                    self.value[state] = 1
                case Player.O:
                    self.value[state] = -1
                case _:
                    raise "Unexpexted winner"

            valid_moves = state.get_valid_moves()
            for valid_move in valid_moves:
                new_state = state.copy()
                new_state.make_move(valid_move[0], valid_move[1])
                open_states.add(new_state)

    def update_value(self, state: TicTacToe, state_prime: TicTacToe, alpha: float):
        self.value[state] = self.value[state] + alpha * (
            self.value[state_prime] - self.value[state]
        )

    def get_value(self, state: TicTacToe) -> float:
        return self.value[state]


if __name__ == "__main__":
    agent = Agent()
    random_player = Agent(exploration=1, exploration_decay=1)

    games_played = 0
    while True:
        game = TicTacToe()
        while not game.is_game_over():
            move = agent.get_move(game, game.current_player)
            game.make_move(move[0], move[1])
        loser = Player.X if game.get_winner() == Player.O else Player.O
        agent.lost_game(game, loser)

        games_played += 1
        if games_played % 100 == 0:
            print(games_played)
            wins = 0
            losses = 0
            for _ in range(100):
                game = TicTacToe()
                current_player = agent
                while not game.is_game_over():
                    move = current_player.get_move(game, game.current_player)
                    game.make_move(move[0], move[1])
                    if current_player == agent:
                        current_player = random_player
                    else:
                        current_player = agent
                wins += game.winner == Player.X
                losses += game.winner == Player.O
            print(f"{wins} vs {losses}")

        if games_played == 3000:
            break

    while True:
        game = TicTacToe()
        while not game.is_game_over():
            if game.current_player == Player.X:
                move = agent.get_move(game, game.current_player)
            else:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                move = (row, col)
            game.make_move(move[0], move[1])
            game.display()
