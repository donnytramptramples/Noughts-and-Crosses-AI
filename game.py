# game.py
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=" ")
            for j in range(3):
                cell_value = self.board[i][j]
                if cell_value == 1:
                    print("X", end=" ")
                elif cell_value == -1:
                    print("O", end=" ")
                else:
                    print(i * 3 + j + 1, end=" ")
                print("|", end=" ")
            print("\n-------------")

    def is_winner(self, player):
        # Check rows, columns, and diagonals
        return any(np.all(self.board == player, axis=0) |  # Rows
                   np.all(self.board == player, axis=1) |  # Columns
                   np.all(np.diag(self.board) == player) |  # Diagonal
                   np.all(np.diag(np.fliplr(self.board)) == player))  # Anti-Diagonal

    def is_full(self):
        return not any(0 in row for row in self.board)

    def is_game_over(self):
        return self.is_winner(1) or self.is_winner(-1) or self.is_full()

    def play_move(self, position):
        row, col = divmod(position - 1, 3)
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False
