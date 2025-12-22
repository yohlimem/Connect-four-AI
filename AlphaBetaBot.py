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
        
        # Move Ordering: Center first
        center = self.COLUMN_COUNT // 2
        valid_locations.sort(key=lambda x: abs(x - center))
        
        # Minimax with Alpha-Beta Pruning
        # Alpha: Best option for maximizer found so far
        # Beta: Best option for minimizer found so far
        col, minimax_score = self.minimax(board, self.depth, -np.inf, np.inf, True, piece)
        
        if col is None:
            col = valid_locations[0] if valid_locations else 0
            
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
                    return (None, 100000000000000)
                elif winning_move_fast(board, -piece): # Opponent wins
                    return (None, -100000000000000)
                else: # Game is over, no more valid moves (Draw)
                    return (None, 0)
            else: # Depth is zero
                return (None, score_position_fast(board, piece))

        if maximizingPlayer:
            value = -np.inf
            column = valid_locations[0]
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
            column = valid_locations[0]
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

class SolvedBot:
    """
    A high-performance bot using Bitboards and Negamax with Alpha-Beta pruning.
    It attempts to calculate the 'perfect' move by searching deep into the game tree.
    """
    def __init__(self):
        self.transposition_table = {}
        # Center columns are usually best in Connect 4
        self.column_order = [3, 2, 4, 1, 5, 0, 6]

    def get_action(self, board: np.ndarray, piece: int) -> int:
        # 1. Parse Board to Bitboard
        # Connect4Env: Row 0 is top, Row 5 is bottom.
        # Bitboard: Col-major. Col 0 (bits 0-5), Col 1 (bits 7-12)...
        position = 0 # Current player's pieces
        mask = 0     # All pieces
        moves_played = 0
        
        for c in range(7):
            for r in range(6):
                if board[r][c] != 0:
                    moves_played += 1
                    # Map row r (0=top, 5=bottom) to bit index
                    # We want bottom (5) to be LSB of the column
                    idx = (5 - r) + c * 7
                    mask |= (1 << idx)
                    if board[r][c] == piece:
                        position |= (1 << idx)

        # 2. Determine Search Depth
        # Bitboards are fast, so we can go deeper than standard bots.
        if moves_played < 10: depth = 10
        elif moves_played < 20: depth = 12
        else: depth = 18 # Attempt to solve endgame

        # 3. Root Search
        best_score = -float('inf')
        best_move = None
        
        valid_moves = [c for c in self.column_order if (mask & (1 << (5 + 7*c))) == 0]
        
        if not valid_moves: return 0

        # Optimization: Check immediate wins first
        for col in valid_moves:
            move = (mask + (1 << (7*col))) & ~mask
            if self.check_win(position | move):
                return col

        alpha = -float('inf')
        beta = float('inf')
        
        for col in valid_moves:
            move = (mask + (1 << (7*col))) & ~mask
            new_pos = position | move
            new_mask = mask | move
            
            # Opponent's turn: Opponent pos is (new_mask ^ new_pos)
            score = -self.negamax(new_mask ^ new_pos, new_mask, depth - 1, -beta, -alpha)
            
            if score > best_score:
                best_score = score
                best_move = col
            
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        
        return best_move if best_move is not None else valid_moves[0]

    def negamax(self, position, mask, depth, alpha, beta):
        key = position | (mask << 49)
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry['depth'] >= depth:
                return entry['value']

        if depth == 0: 
            return self.evaluate(position, mask)
            
        if mask == 0x1FBFFFFFFFFFF: return 0 # Draw (Board full)

        max_val = -float('inf')
        
        for col in self.column_order:
             if (mask & (1 << (5 + 7*col))) == 0:
                move = (mask + (1 << (7*col))) & ~mask
                new_pos = position | move
                
                if self.check_win(new_pos):
                    val = 1000 + depth # Prefer faster wins
                else:
                    new_mask = mask | move
                    val = -self.negamax(new_mask ^ new_pos, new_mask, depth - 1, -beta, -alpha)
                
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if alpha >= beta: break
        
        self.transposition_table[key] = {'value': max_val, 'depth': depth}
        return max_val

    def evaluate(self, position, mask):
        """
        Heuristic evaluation for non-terminal nodes.
        Prioritizes center control, which is critical in Connect 4.
        """
        score = 0
        
        # Center Column (Col 3) bits: 21, 22, 23, 24, 25, 26
        # 0x3F << 21
        center_mask = 0x7E00000 
        
        my_center = bin(position & center_mask).count('1')
        
        opp_position = position ^ mask
        opp_center = bin(opp_position & center_mask).count('1')
        
        score += (my_center * 3)
        score -= (opp_center * 3)
        
        return score

    def check_win(self, pos):
        # Bitwise win check for all directions
        m = pos & (pos >> 7);  # Horizontal
        if m & (m >> 14): return True
        m = pos & (pos >> 1);  # Vertical
        if m & (m >> 2): return True
        m = pos & (pos >> 6);  # Diagonal 1
        if m & (m >> 12): return True
        m = pos & (pos >> 8);  # Diagonal 2
        if m & (m >> 16): return True
        return False