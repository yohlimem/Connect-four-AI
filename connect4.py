import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any

import torch
from policy import Policy
from state_value import StateValue 

class Connect4Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the arena where The Sigma's AI will train.
    """
    def __init__(self) -> None:
        super(Connect4Env, self).__init__()
        # 6 rows, 7 columns
        self.rows: int = 6
        self.cols: int = 7
        
        # Action space: 0-6 (Drop in one of the columns)
        self.action_space = spaces.Discrete(self.cols)
        
        # Observation space: 6x7 grid. 
        # 0 = Empty, 1 = AI (Sigma), 2 = Opponent
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int8)
        
        self.board: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player: int = 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1 # AI always starts for now
        return self.board, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # The action is for the current_player
        is_agent_turn: bool = self.current_player == 1

        # 1. Check if move is valid (Column not full)
        if self.board[0][action] != 0:
            # Invalid move for the current player.
            return self.board, -10, True, False, {"error": "Invalid Move", "winner": None}

        # 2. Execute Move (Gravity)
        row: int = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        # 3. Check Win Condition for the current player
        if self.check_win(self.current_player):
            reward: float = 10 if is_agent_turn else -10
            return self.board, 10, True, False, {"info": f"Player {self.current_player} wins!", "winner": self.current_player}

        # 4. Check Draw
        if np.all(self.board != 0):
            return self.board, 0, True, False, {"info": f"It a draw!", "winner": None} # Draw

        # 5. If the game is not over, it's the next player's turn.
        self.switch_player()
        
        # The reward for a non-terminal move.
        return self.board, 0, False, False, {}

    def get_next_open_row(self, col: int) -> int:
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def switch_player(self) -> None:
        self.current_player = -1 if self.current_player == 1 else 1

    def check_win(self, piece: int) -> bool:
        # Check horizontal locations for win
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True
        return False

    def render(self, mode: str = 'human') -> None:
        print(self.board)

if __name__ == '__main__':

    env = Connect4Env()
    ppo_policy = Policy(StateValue(env.observation_space.shape[0] * env.observation_space.shape[1], 3, 256), env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n, 3, 256) # pyright: ignore[reportOptionalSubscript]

    for i in range(100000):
        reward = 0 # Initialize reward for the game over message
        obs, info = env.reset()
        terminated = False
        # env.render()
        # print("-" * 20)
        observations = []
        actions = []
        rewards = []
        old_probs = []
        while not terminated:
            observations.append(obs.flatten())
            with torch.no_grad():
                action_probs = ppo_policy(torch.from_numpy(obs*env.current_player).float().flatten().unsqueeze(0))[0] # the index of the best probabilty is also the action number
                action = torch.argmax(action_probs).item() # the index of the best probabilty is also the action number
                old_probs.append(action_probs)
                actions.append(action)

                

            
            # print(f"Player {env.current_player} chooses column {action}")

            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(reward)

            
            # env.render()
            # print(f"Reward: {reward}")
            # print("-" * 20)

        observations_np = np.array(observations, dtype="float32")
        old_probs_np = np.array(old_probs, dtype="float32")

        observations_tensor = torch.from_numpy(observations_np)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        old_probs_tensor = torch.from_numpy(old_probs_np)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantage = ppo_policy.advantage(observations_tensor, rewards_tensor)
        

        # print("final board:")
        # print(env.board)

        # print(observations)
        # print("actions:")
        # print(actions)
        if i  % 100 == 0:
            print(f"rewards at epoch {i}:")
            print(rewards)
            print(advantage)
        # print("old_probs:")
        # print(len(old_probs))
        # print(len(actions))
        # print(len(observations))
        # print("advantage:")
        # print(advantage)
        for _ in range(1):
            ppo_policy.optimizer_step(observations_tensor, actions_tensor, old_probs_tensor, rewards_tensor, advantage)
            


    