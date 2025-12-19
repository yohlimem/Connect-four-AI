import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import torch
from torch.distributions import Categorical
import torch.utils.data as data_utils
import matplotlib.pyplot as plt

# Import your classes
from policy import Policy
from state_value import StateValue
from sigma_test import test_sigma_agent

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        is_agent_turn: bool = self.current_player == 1

        if self.board[0][action] != 0:
            self.switch_player()
            return self.board, -0.1, False, False, {"error": "Invalid Move", "winner": None}

        row: int = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        if self.check_win(self.current_player):
            reward: float = 1.0 if is_agent_turn else -1.0 # Standardize to 1/-1
            return self.board, reward, True, False, {"info": f"Player {self.current_player} wins!", "winner": self.current_player}

        if np.all(self.board != 0):
            return self.board, 0, True, False, {"info": f"It a draw!", "winner": None}

        self.switch_player()
        return self.board, 0, False, False, {}

    def get_next_open_row(self, col: int) -> int:
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def switch_player(self) -> None:
        self.current_player = -1 if self.current_player == 1 else 1

    def check_win(self, piece: int) -> bool:
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True
        return False

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

def preprocess_board(board, current_player):
    """
    Converts a (6, 7) board with [-1, 0, 1] 
    into a (2, 6, 7) tensor with [0, 1].
    
    Channel 0: My Pieces
    Channel 1: Enemy Pieces
    """
    # If current_player is -1, we flip the board signs so the AI always sees itself as '1'
    relative_board = board * current_player
    
    # Masks
    my_pieces = (relative_board == 1).astype(np.float32)
    enemy_pieces = (relative_board == -1).astype(np.float32)
    
    # Stack them
    # Result shape: (2, 6, 7)
    return np.stack([my_pieces, enemy_pieces])


if __name__ == '__main__':
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    env = Connect4Env()

    state_value = StateValue(input_channels=2, board_height=6, board_width=7)
    ppo_policy = Policy(state_value, env.action_space.n, input_channels=2, board_height=6, board_width=7, ent_coef=0.03)

    # Optional: Load previous weights if architecture matches
    # ppo_policy.load_from_file("connect4_policy_iter_600000.pth")

    total_observations = []
    total_actions = []
    total_rewards = []
    total_old_probs = []
    total_advantages = []
    objectives = []
    loss = []

    for i in range(1):
        obs, info = env.reset()
        terminated = False
        
        observations = []
        actions = []
        rewards = []
        old_probs = []
        steps = 0

        while not terminated and steps < 6*8:
            steps += 1
            
            # Do NOT flatten here. Multiply by current player for perspective.
            current_obs = preprocess_board(obs, env.current_player)
            observations.append(current_obs)

            with torch.no_grad():
                # Prepare input: (1, 2, 6, 7) for CNN
                tensor_input = torch.from_numpy(current_obs).unsqueeze(0).float()
                
                unmasked_probs = ppo_policy(tensor_input)[0]

                # Masking
                legal_moves_mask = torch.tensor([0.0 if env.board[0][c] == 0 else -1e9 for c in range(env.cols)])
                masked_probs = unmasked_probs + legal_moves_mask
                
                dist = Categorical(logits=masked_probs)
                action = dist.sample()

                actions.append(action.item())
                old_probs.append(dist.log_prob(action))

            obs, _, terminated, _, info = env.step(int(action))

        # Reward Calculation
        gamma = 0.99
        discounts = [gamma**i for i in range(len(observations))][::-1] 

        if info["winner"] == 1:
            rewards = [(1 * discounts[i]) if i % 2 == 0 else (-1 * discounts[i]) for i in range(len(observations))]
        elif info["winner"] == -1:
            rewards = [(-1 * discounts[i]) if i % 2 == 0 else (1 * discounts[i]) for i in range(len(observations))]
        else:
            rewards = [0] * len(observations)

        # Batch Preparation
        observations_np = np.array(observations, dtype="float32") # Shape (N, 2, 6, 7)
        old_probs_np = np.array(old_probs, dtype="float32")

        observations_tensor = torch.from_numpy(observations_np)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        old_probs_tensor = torch.from_numpy(old_probs_np)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Advantage
        advantage = ppo_policy.advantage(observations_tensor, rewards_tensor)
        
        # Batches for training
        total_observations.append(observations_tensor)
        total_actions.append(actions_tensor)
        total_rewards.append(rewards_tensor)
        total_old_probs.append(old_probs_tensor)
        total_advantages.append(advantage)

        # Logging
        if i % 100 == 0:
            objectives.append(ppo_policy.objective(observations_tensor, actions_tensor, old_probs_tensor, rewards_tensor, advantage).detach().item())
            loss.append(ppo_policy.value_function.loss(observations_tensor, rewards_tensor).detach().item())

        if i % 10000 == 0:
            print("final board:")
            env.render()
            print(f"at epoch {i}:")
            print("obj:", objectives[-1] if objectives else 0)
            print("loss:", loss[-1] if loss else 0)
            # check against a random bot!
            test_sigma_agent(Connect4Env, policy=ppo_policy, games=100) # Uncomment if file exists

        # Optimization
        if i % 1000 == 0 and i > 0:
            # Concatenate list of tensors to create a database
            dataset = data_utils.TensorDataset(
                torch.cat(total_observations), 
                torch.cat(total_actions), 
                torch.cat(total_old_probs), 
                torch.cat(total_rewards), 
                torch.cat(total_advantages)
            )
            loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)
            
            for _ in range(5):
                for batch_obs, batch_action, batch_old_prob, batch_reward, batch_advantage in loader:
                    ppo_policy.optimizer_step(batch_obs, batch_action, batch_old_prob, batch_reward, batch_advantage)
                
            total_observations = []
            total_actions = []
            total_rewards = []
            total_old_probs = []
            total_advantages = []
        
        # Saving (by gemini)
        if i > 0 and i % 100000 == 0:
            save_path = f"connect4_cnn_iter_{i}.pth"
            torch.save(ppo_policy.state_dict(), save_path)
            print(f"Model state saved to {save_path}")

        if i > 0 and i % 50000 == 0:
            plt.figure()
            plt.plot(objectives)
            plt.title(f"Objective at step {i}")
            plt.savefig(f"training_graph_{i}.png")
            plt.close()
            print(f"Graph saved to training_graph_{i}.png")