from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class Player(Enum):
    """Represents the players in the game"""

    X = 1
    O = -1
    EMPTY = 0


class TicTacToe:
    """
    Tic-Tac-Toe game implementation for reinforcement learning.

    The board is represented as a 3x3 numpy array where:
    - 1 represents Player X
    - -1 represents Player O
    - 0 represents an empty cell
    """

    def __init__(self):
        """Initialize an empty tic-tac-toe board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_over = False
        self.winner = None
        self.move_history = []

    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.

        Returns:
            The initial board state
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get the current board state.

        Returns:
            A copy of the current board state
        """
        return self.board.copy()

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Get all valid moves (empty cells) on the board.

        Returns:
            List of (row, col) tuples representing valid moves
        """
        valid_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == Player.EMPTY.value:
                    valid_moves.append((row, col))
        return valid_moves

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if a move is valid.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            True if the move is valid, False otherwise
        """
        if self.game_over:
            return False
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        return self.board[row, col] == Player.EMPTY.value

    def make_move(self, row: int, col: int) -> Tuple[bool, Optional[int]]:
        """
        Make a move on the board.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)

        Returns:
            Tuple of (success, reward):
            - success: True if move was valid and executed
            - reward: 1 if current player wins, -1 if opponent wins,
                      0 if draw, None if game continues
        """
        if not self.is_valid_move(row, col):
            return False, None

        # Make the move
        self.board[row, col] = self.current_player.value
        self.move_history.append((row, col, self.current_player))

        # Check for win or draw
        reward = self._check_game_over()

        # Switch player if game is not over
        if not self.game_over:
            self.current_player = (
                Player.O if self.current_player == Player.X else Player.X
            )

        return True, reward

    def _check_game_over(self) -> Optional[int]:
        """
        Check if the game is over (win or draw).

        Returns:
            Reward value: 1 if X wins, -1 if O wins, 0 if draw, None if game continues
        """
        # Check rows
        for row in range(3):
            if abs(self.board[row].sum()) == 3:
                self.game_over = True
                self.winner = Player.X if self.board[row, 0] == 1 else Player.O
                return 1 if self.winner == Player.X else -1

        # Check columns
        for col in range(3):
            if abs(self.board[:, col].sum()) == 3:
                self.game_over = True
                self.winner = Player.X if self.board[0, col] == 1 else Player.O
                return 1 if self.winner == Player.X else -1

        # Check main diagonal
        main_diag_sum = np.trace(self.board)
        if abs(main_diag_sum) == 3:
            self.game_over = True
            self.winner = Player.X if main_diag_sum == 3 else Player.O
            return 1 if self.winner == Player.X else -1

        # Check anti-diagonal
        anti_diag_sum = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        if abs(anti_diag_sum) == 3:
            self.game_over = True
            self.winner = Player.X if anti_diag_sum == 3 else Player.O
            return 1 if self.winner == Player.X else -1

        # Check for draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = None
            return 0

        return None

    def get_current_player(self) -> Player:
        """Get the current player"""
        return self.current_player

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_over

    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game (None if draw or game not over)"""
        return self.winner

    def display(self):
        """Display the current board state"""
        symbols = {1: "X", -1: "O", 0: " "}
        print("\n  0   1   2")
        print("  ---------")
        for row in range(3):
            print(
                f"{row}| {symbols[self.board[row, 0]]} | {symbols[self.board[row, 1]]} | {symbols[self.board[row, 2]]} |"
            )
            if row < 2:
                print("  ---------")
        print()

    def __hash__(self):
        return hash(self.get_state_hash())
    
    def __eq__(self, value):
        if not isinstance(value, TicTacToe):
            False
        
        return self.get_state_hash() == value.get_state_hash()

    def get_state_hash(self) -> str:
        """
        Get a hash string representation of the board state.
        Useful for RL algorithms that use state lookup tables.

        Returns:
            String representation of the board state
        """
        return str(self.board.flatten())

    def copy(self) -> "TicTacToe":
        """
        Create a deep copy of the game state.

        Returns:
            A new TicTacToe instance with the same state
        """
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        return new_game


# Example usage and testing
if __name__ == "__main__":
    # Create a game instance
    game = TicTacToe()

    # Example game play
    print("Tic-Tac-Toe Game")
    print("=" * 30)

    # Play a sample game
    moves = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]  # X wins

    for move in moves:
        row, col = move
        print(f"Player {game.get_current_player().name} plays at ({row}, {col})")
        success, reward = game.make_move(row, col)
        if success:
            game.display()

            if game.is_game_over():
                if game.get_winner():
                    print(f"Player {game.get_winner().name} wins!")
                else:
                    print("It's a draw!")
                break

    # Reset and show valid moves
    print("\n" + "=" * 30)
    print("Resetting game...")
    game.reset()
    print("Valid moves:", game.get_valid_moves())
    game.display()
