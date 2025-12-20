'''
Docstring for AlphaBetaBot

! MADE WITH GEMINI TO CHECK MY OWN AI BOT AGAINST !
'''


import numpy as np
import random

class AlphaBetaBot:
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.WINDOW_LENGTH = 4

    def get_action(self, board: np.ndarray, piece: int) -> int:
        """
        Main entry point. Returns the best column (0-6).
        'piece' should be the bot's player ID (e.g., 1 or -1).
        """
        # Get valid locations first to save time
        valid_locations = self.get_valid_locations(board)
        
        # Minimax with Alpha-Beta Pruning
        # Alpha: Best option for maximizer found so far
        # Beta: Best option for minimizer found so far
        col, minimax_score = self.minimax(board, self.depth, -np.inf, np.inf, True, piece)
        
        if col is None:
            col = random.choice(valid_locations)
            
        return col

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, piece):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board, piece)
        
        # Base Case: Terminal Node or Max Depth
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, piece):
                    return (None, 100000000000000)
                elif self.winning_move(board, -piece): # Opponent wins
                    return (None, -100000000000000)
                else: # Game is over, no more valid moves (Draw)
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(board, piece))

        if maximizingPlayer:
            value = -np.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                b_copy[row][col] = piece
                new_score = self.minimax(b_copy, depth-1, alpha, beta, False, piece)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else: # Minimizing Player
            value = np.inf
            column = random.choice(valid_locations)
            opponent_piece = -piece
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                b_copy[row][col] = opponent_piece
                new_score = self.minimax(b_copy, depth-1, alpha, beta, True, piece)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def score_position(self, board, piece):
        score = 0
        opponent_piece = -piece

        # Score Center Column (Prefer center)
        center_array = [int(i) for i in list(board[:, self.COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(self.COLUMN_COUNT-3):
                window = row_array[c:c+self.WINDOW_LENGTH]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Score Vertical
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(self.ROW_COUNT-3):
                window = col_array[r:r+self.WINDOW_LENGTH]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Score Positive Sloped Diagonal
        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Score Negative Sloped Diagonal
        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                window = [board[r+3-i][c+i] for i in range(self.WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece, opponent_piece)

        return score

    def evaluate_window(self, window, piece, opponent_piece):
        score = 0
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opponent_piece) == 3 and window.count(0) == 1:
            score -= 4 # Block opponent!

        return score

    def is_terminal_node(self, board, piece):
        return self.winning_move(board, piece) or self.winning_move(board, -piece) or len(self.get_valid_locations(board)) == 0

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.COLUMN_COUNT):
            if board[0][col] == 0:
                valid_locations.append(col)
        return valid_locations

    def get_next_open_row(self, board, col):
        for r in range(self.ROW_COUNT-1, -1, -1):
            if board[r][col] == 0:
                return r
        return None # Should not happen if called on valid col

    def winning_move(self, board, piece):
        # Helper to reuse the win logic without modifying the env
        # Horizontal
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        # Vertical
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        # Pos Slope
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        # Neg Slope
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False