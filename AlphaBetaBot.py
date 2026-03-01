'''
Docstring for AlphaBetaBot

! MADE WITH GEMINI TO CHECK MY OWN AI BOT AGAINST !
'''


import numpy as np
import random
from numba import njit

@njit
def get_next_open_row_fast(board, col):
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return -1

@njit
def winning_move_fast(board, piece):
    ROWS = 6
    COLS = 7
    # Horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r, c] == piece and board[r, c+1] == piece and board[r, c+2] == piece and board[r, c+3] == piece:
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r, c] == piece and board[r+1, c] == piece and board[r+2, c] == piece and board[r+3, c] == piece:
                return True
    # Pos Slope
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r, c] == piece and board[r+1, c+1] == piece and board[r+2, c+2] == piece and board[r+3, c+3] == piece:
                return True
    # Neg Slope
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r, c] == piece and board[r-1, c+1] == piece and board[r-2, c+2] == piece and board[r-3, c+3] == piece:
                return True
    return False

@njit
def score_position_fast(board, piece):
    score = 0
    opponent_piece = -piece
    ROWS = 6
    COLS = 7
    
    # Center Column Preference
    center_col = COLS // 2
    for r in range(ROWS):
        if board[r, center_col] == piece:
            score += 3

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            p, o, e = 0, 0, 0
            for i in range(4):
                val = board[r, c+i]
                if val == piece: p += 1
                elif val == opponent_piece: o += 1
                else: e += 1
            if p == 4: score += 100
            elif p == 3 and e == 1: score += 5
            elif p == 2 and e == 2: score += 2
            if o == 3 and e == 1: score -= 4

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            p, o, e = 0, 0, 0
            for i in range(4):
                val = board[r+i, c]
                if val == piece: p += 1
                elif val == opponent_piece: o += 1
                else: e += 1
            if p == 4: score += 100
            elif p == 3 and e == 1: score += 5
            elif p == 2 and e == 2: score += 2
            if o == 3 and e == 1: score -= 4

    # Pos Slope
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p, o, e = 0, 0, 0
            for i in range(4):
                val = board[r+i, c+i]
                if val == piece: p += 1
                elif val == opponent_piece: o += 1
                else: e += 1
            if p == 4: score += 100
            elif p == 3 and e == 1: score += 5
            elif p == 2 and e == 2: score += 2
            if o == 3 and e == 1: score -= 4

    # Neg Slope
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            p, o, e = 0, 0, 0
            for i in range(4):
                val = board[r+3-i, c+i]
                if val == piece: p += 1
                elif val == opponent_piece: o += 1
                else: e += 1
            if p == 4: score += 100
            elif p == 3 and e == 1: score += 5
            elif p == 2 and e == 2: score += 2
            if o == 3 and e == 1: score -= 4
            
    return score

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
        if not valid_locations:
            return 0 # Should not happen in a normal game

        # Move Ordering: Center first
        center = self.COLUMN_COUNT // 2
        valid_locations.sort(key=lambda x: abs(x - center))
        
        # Minimax with Alpha-Beta Pruning
        col, minimax_score = self.minimax(board, self.depth, -np.inf, np.inf, True, piece)
        
        if col is None:
            return random.choice(valid_locations)
            
        return col

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, piece):
        valid_locations = self.get_valid_locations(board)
        
        # Move Ordering
        center = self.COLUMN_COUNT // 2
        valid_locations.sort(key=lambda x: abs(x - center))
        
        is_terminal = winning_move_fast(board, piece) or winning_move_fast(board, -piece) or len(valid_locations) == 0
        
        # Base Case: Terminal Node or Max Depth
        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move_fast(board, piece):
                    return (None, 100000000000000 + depth)
                elif winning_move_fast(board, -piece): # Opponent wins
                    # print("already lost :(")
                    return (None, -100000000000000 - depth)
                else: # Game is over, no more valid moves (Draw)
                    return (None, 0)
            else: # Depth is zero
                return (None, score_position_fast(board, piece))

        if maximizingPlayer:
            value = -np.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row_fast(board, col)
                # No Copy Optimization
                board[row][col] = piece
                new_score = self.minimax(board, depth-1, alpha, beta, False, piece)[1]
                board[row][col] = 0 # Undo
                
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
                row = get_next_open_row_fast(board, col)
                # No Copy Optimization
                board[row][col] = opponent_piece
                new_score = self.minimax(board, depth-1, alpha, beta, True, piece)[1]
                board[row][col] = 0 # Undo
                
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.COLUMN_COUNT):
            if board[0][col] == 0:
                valid_locations.append(col)
        return valid_locations