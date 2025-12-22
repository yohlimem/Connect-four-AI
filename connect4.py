import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import torch
from torch.distributions import Categorical
import torch.utils.data as data_utils
import matplotlib.pyplot as plt

# Import your classes
from AlphaBetaBot import AlphaBetaBot
from policy import Policy
from state_value import StateValue

import sys

from numba import njit


@njit
def check_win_fast(board, row, col, piece):
    # Constants for clarity
    ROWS = 6
    COLS = 7

    # 1. Check Horizontal (-)
    count = 0
    # Optimization: We only need to check the row 'row'
    for c in range(COLS):
        if board[row][c] == piece:
            count += 1
            if count == 4: return True
        else:
            count = 0

    # 2. Check Vertical (|)
    count = 0
    # Optimization: We only need to check the col 'col'
    for r in range(ROWS):
        if board[r][col] == piece:
            count += 1
            if count == 4: return True
        else:
            count = 0
    
    # 3. Check Main Diagonal (\) 
    # Direction: Top-Left to Bottom-Right
    # We find the Top-Left-most start point
    count = 0
    offset = min(row, col)
    r = row - offset
    c = col - offset
    
    while r < ROWS and c < COLS:
        if board[r][c] == piece:
            count += 1
            if count == 4: return True
        else:
            count = 0
        r += 1
        c += 1

    # 4. Check Anti-Diagonal (/) 
    # Direction: Top-Right to Bottom-Left
    # We find the Top-Right-most start point
    count = 0
    
    # FIX IS HERE: Distance to right edge is (COLS - 1) - col
    offset = min(row, (COLS - 1) - col) 
    
    r = row - offset
    c = col + offset
    
    while r < ROWS and c >= 0:
        if board[r][c] == piece:
            count += 1
            if count == 4: return True
        else:
            count = 0
        r += 1
        c -= 1
        
    return False

@njit
def get_next_open_row_fast(board, col):
    for r in range(5, -1, -1): # 6 rows: 5 to 0
        if board[r][col] == 0:
            return r
    return -1

class Connect4Env(gym.Env):
    def __init__(self) -> None:
        super(Connect4Env, self).__init__()
        self.rows: int = 6
        self.cols: int = 7
        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int8)
        self.board: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player: int = 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1 
        return self.board, {}

    def step(self, action):
        # Quick check for invalid move
        # print("hi")
        if self.board[0][action] != 0:
            return self.board.copy(), -0.1, False, False, {"error": "Invalid", "winner": 0}

        # Fast Row Lookup
        row = get_next_open_row_fast(self.board, action)
        self.board[row][action] = self.current_player

        # Fast Incremental Win Check (Only checks the new piece)
        if check_win_fast(self.board, row, action, self.current_player):
            reward = 1.0 if self.current_player == 1 else -1.0
            return self.board.copy(), reward, True, False, {"winner": self.current_player}

        # Draw Check
        if self.board[0].all(): # If top row is full, board is full
            return self.board.copy(), 0, True, False, {"winner": 0}

        self.current_player *= -1 # Fast switch (1 -> -1, -1 -> 1)
        return self.board.copy(), 0, False, False, {"winner": 0}

    def get_next_open_row(self, col: int) -> int:
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def switch_player(self) -> None:
        self.current_player = -1 if self.current_player == 1 else 1

    def render(self, mode: str = 'human') -> None:
        colors = {0: "\x1b[40m", 1: "\x1b[44m", -1: "\x1b[41m"}
        reset = "\x1b[0m"
        for row in self.board:
            print("".join(f"{colors.get(cell, reset)}  {reset}" for cell in row))
        print(reset)

def render_board(board: np.ndarray) -> None:
    colors = {0: "\x1b[40m", 1: "\x1b[44m", -1: "\x1b[41m"}
    reset = "\x1b[0m"
    for row in board:
        print("".join(f"{colors.get(cell, reset)}  {reset}" for cell in row))
    print(reset)








