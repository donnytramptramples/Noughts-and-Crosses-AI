# main.py
from game import TicTacToe
from model import TicTacToeAI
from genetic_algorithm import evaluate_model  # Add this import

def play_game():
    game = TicTacToe()
    ai_model = TicTacToeAI()

    while not game.is_game_over():
        game.print_board()

        if game.current_player == 1:  # Human's turn
            try:
                position = int(input("Enter position (1-9): "))
                if 1 <= position <= 9:
                    if game.play_move(position):
                        print("Move accepted!\n")
                    else:
                        print("Invalid move. Try again.\n")
                else:
                    print("Invalid input. Please enter a number between 1 and 9.\n")
            except ValueError:
                print("Invalid input. Please enter a valid number.\n")
        else:  # AI's turn
            # AI makes a move based on its learned strategy
            best_move = get_best_move(ai_model, game)
            game.play_move(best_move)
            print("AI made a move!\n")

    game.print_board()

    if game.is_winner(1):
        print("You win!")
    elif game.is_winner(-1):
        print("AI wins!")
    else:
        print("It's a draw!")

def get_best_move(model, game):
    # Evaluate all possible moves and return the best one
    best_score = float('-inf')
    best_move = None

    for i in range(1, 10):
        row, col = divmod(i - 1, 3)
        if game.board[row][col] == 0:
            game.board[row][col] = game.current_player
            score = evaluate_model(model, game.board)
            game.board[row][col] = 0  # Reset the move

            if score > best_score:
                best_score = score
                best_move = i

    return best_move

if __name__ == "__main__":
    play_game()
