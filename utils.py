import numpy as np
import torch

def preprocess_board(board, current_player=None):
    """
    Converts a (6, 7) board with [-1, 0, 1] 
    into a (2, 6, 7) tensor with [0, 1].
    Given batches of boards it infers the current player.
    
    Channel 0: My Pieces
    Channel 1: Enemy Pieces
    """
    # Handle Batch Input (N, 6, 7)
    if board.ndim == 3:
        # Infer player: sum=0 -> P1(1), sum=1 -> P2(-1)
        sums = board.sum(axis=(1,2))
        players = np.where(sums == 0, 1, -1).reshape(-1, 1, 1)
        relative_board = board * players
        my_pieces = (relative_board == 1).astype(np.float32)
        enemy_pieces = (relative_board == -1).astype(np.float32)
        return np.stack([my_pieces, enemy_pieces], axis=1)

    # If current_player is -1, we flip the board signs so the AI always sees itself as '1'
    if current_player is None:
        current_player = 1 if np.sum(board) == 0 else -1
    relative_board = board * current_player
    
    # Masks
    my_pieces = (relative_board == 1).astype(np.float32)
    enemy_pieces = (relative_board == -1).astype(np.float32)
    
    # Stack them
    # Result shape: (2, 6, 7)
    return np.stack([my_pieces, enemy_pieces])