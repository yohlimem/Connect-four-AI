import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any

import torch
from policy import Policy
from state_value import StateValue 

from torch.distributions import Categorical
import torch.utils.data as data_utils

import matplotlib.pyplot as plt



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
            self.switch_player()
            return self.board, -0.1, False, False, {"error": "Invalid Move", "winner": None}

        # 2. Execute Move (Gravity)
        row: int = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        # 3. Check Win Condition for the current player
        if self.check_win(self.current_player):
            reward: float = 10 if is_agent_turn else -10
            return self.board, 1, True, False, {"info": f"Player {self.current_player} wins!", "winner": self.current_player}

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
        # ANSI escape codes for backgrounds
        colors = {
            0: "\x1b[40m",  # Black for empty
            1: "\x1b[44m",  # Blue for Player 1
            -1: "\x1b[41m", # Red for Player -1
        }
        reset = "\x1b[0m"

        if self.board is None:
            print("Board is not initialized.")
            return

        for row in self.board:
            # For each cell, print the color, two spaces for visibility, then reset
            print("".join(f"{colors.get(cell, reset)}  {reset}" for cell in row))
        print(reset) # Ensure color is reset at the end of printing


def render(board: np.ndarray) -> None:
    # ANSI escape codes for backgrounds
    colors = {
        0: "\x1b[40m",  # Black for empty
        1: "\x1b[44m",  # Blue for Player 1
        -1: "\x1b[41m", # Red for Player -1
    }
    reset = "\x1b[0m"

    for row in board:
        # For each cell, print the color, two spaces for visibility, then reset
        print("".join(f"{colors.get(cell, reset)}  {reset}" for cell in row))
    print(reset) # Ensure color is reset at the end of printing
if __name__ == '__main__':
    
    from sigma_test import test_sigma_agent

    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))



    env = Connect4Env()
    ppo_policy = Policy(StateValue(env.observation_space.shape[0] * env.observation_space.shape[1], 3, 256), env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n, 3, 256, ent_coef=0.03) # pyright: ignore[reportOptionalSubscript]

    ppo_policy.load_from_file("connect4_policy_iter_600000.pth")


    total_observations = []
    total_actions = []
    total_rewards = []
    total_old_probs = []
    total_advantages = []
    objectives = []
    loss = []

    for i in range(10000000):
        reward = 0 # Initialize reward for the game over message
        obs, info = env.reset()
        terminated = False
        # env.render()
        # print("-" * 20)
        observations = []
        actions = []
        rewards = []
        old_probs = []

        steps = 0
        while not terminated and steps < 6*8: # a little bigger than the size of the board to let the ai expirament
            steps+=1
            observations.append(obs.flatten()*env.current_player)
            with torch.no_grad():
                # Get action probabilities from the policy
                unmasked_probs = ppo_policy(torch.from_numpy(obs*env.current_player).float().flatten().unsqueeze(0))[0]

                legal_moves_mask = torch.tensor([0.0 if env.board[0][c] == 0 else -(10e8) for c in range(env.cols)])


                # Apply the mask to zero out probabilities for illegal moves
                masked_probs = unmasked_probs + legal_moves_mask

                dist = Categorical(logits=masked_probs)

                action = dist.sample()

                actions.append(action.item())
                old_probs.append(dist.log_prob(action))


                # mask for legal moves. A move is legal if the top row of the column is empty (0).

                

            
            # print(f"Player {env.current_player} chooses column {action}")

            obs, _, terminated, truncated, info = env.step(int(action))
            # print("-" * 20)
            # printervations)
            # print("actions:")
            # print(actions[-3:])
            # print("observations:")
            # [render(o.reshape(6,7)) for o in observations[-1:]]


        # Create a discount list [1.0, 0.99, 0.98, ...] reversed
        gamma = 0.99
        discounts = [gamma**i for i in range(len(observations))][::-1] 

        if info["winner"] == 1:
            rewards = [(1 * discounts[i]) if i % 2 == 0 else (-1 * discounts[i]) for i in range(len(observations))]
        elif info["winner"] == -1:
            rewards = [(-1 * discounts[i]) if i % 2 == 0 else (1 * discounts[i]) for i in range(len(observations))]
        else:
            rewards = [0] * len(observations)


        observations_np = np.array(observations, dtype="float32")
        old_probs_np = np.array(old_probs, dtype="float32")

        observations_tensor = torch.from_numpy(observations_np)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        old_probs_tensor = torch.from_numpy(old_probs_np)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantage = ppo_policy.advantage(observations_tensor, rewards_tensor)
        
        total_observations.append(observations_tensor)
        total_actions.append(actions_tensor)
        total_rewards.append(rewards_tensor)
        total_old_probs.append(old_probs_tensor)
        total_advantages.append(advantage)


        # print(observations)
        # print("actions:")
        # print(actions)

        if i % 100 == 0:
            objectives.append(ppo_policy.objective(observations_tensor, actions_tensor, old_probs_tensor, rewards_tensor, advantage).detach().item())
            loss.append(ppo_policy.value_function.loss(observations_tensor, rewards_tensor).detach().item())

            

        if i % 10000 == 0:
            print("final board:")
            for j in range(len(observations_np)):
                render(observations_np[j].reshape(6,7))
            env.render()
            print(f"at epoch {i}:")
            print("rewards:", rewards)
            print("advantage:", advantage)
            print("objective:", objectives[-1])
            print("last objectives:", objectives[-10:])
            print("loss:", loss[-1])
            print("last loss:", loss[-10:])
            test_sigma_agent(Connect4Env, policy=ppo_policy, games=100)


        if i % 1000 == 0 and i > 0:
            dataset = data_utils.TensorDataset(torch.cat(total_observations), torch.cat(total_actions), torch.cat(total_old_probs), torch.cat(total_rewards), torch.cat(total_advantages))
            loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)
            for _ in range(5):
                for batch_obs, batch_action, batch_old_prob, batch_reward, batch_advantage in loader:
                    ppo_policy.optimizer_step(batch_obs, batch_action, batch_old_prob, batch_reward, batch_advantage)
                
            total_observations = []
            total_actions = []
            total_rewards = []
            total_old_probs = []
            total_advantages = []
        
        # print("old_probs:")
        # print(len(old_probs))
        # print(len(actions))
        # print(len(observations))
        # print("advantage:")
        # print(advantage)
        
        if i > 0 and i % 100000 == 0:
            save_path = f"connect4_policy_iter_{i}.pth"
            torch.save(ppo_policy.state_dict(), save_path)
            print(f"Model state saved to {save_path}")
        if i > 0 and i % 50000 == 0:
            plt.figure() # Create a new figure
            plt.plot(objectives)
            plt.title(f"Objective at step {i}")
            plt.xlabel("Steps (x100)")
            plt.ylabel("Objective")
            
            plt.savefig(f"training_graph_{i}.png")
            
            plt.close() # Close the figure to free up memory (Critical!)
            print(f"Graph saved to training_graph_{i}.png")


    