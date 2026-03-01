import numpy as np
import torch
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

from connect4 import Connect4Env
from utils import preprocess_board
from AlphaBetaBot import AlphaBetaBot
from AlphaBetaBot import SolvedBot as AlphaBetaSolvedBot
from SolvedBot import SolvedBot
from policy import Policy

class PolicyBot:
    def __init__(self, model_path):
        self.policy = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256])
        self.policy.load_from_file(model_path)
        self.policy.eval()

    def get_action(self, board, piece):
        ai_input = torch.from_numpy(preprocess_board(board, piece)).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy.forward(ai_input)[0]
            mask = torch.tensor([0.0 if board[0][c] == 0 else -1e9 for c in range(7)])
            action = torch.argmax(logits + mask).item()
        return action

def generate_random_state(n_moves):
    env = Connect4Env()
    obs, _ = env.reset()
    done = False
    move_sequence = []
    for _ in range(n_moves):
        if done:
            break
        legal_moves = [c for c in range(env.cols) if env.board[0][c] == 0]
        if not legal_moves:
            break
        action = random.choice(legal_moves)
        obs, _, done, _, _ = env.step(action)
        move_sequence.append(str(action + 1))
    return obs, env.current_player, "".join(move_sequence)

def run_benchmark():
    # 0. Create directory for saving plots
    save_dir = 'Saves/confusion_matrices'
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialize Bots
    solved_bot = SolvedBot()
    ab_bot = AlphaBetaBot(depth=4)
    ab_solved_bot = AlphaBetaSolvedBot()
    policy_bot = PolicyBot('This bot is super good large CNN.pth')

    bots = {"AlphaBetaBot": ab_bot, "AlphaBetaSolvedBot": ab_solved_bot, "PolicyBot": policy_bot}
    results = {name: [] for name in bots.keys()}
    ground_truth = []

    # 2. Generate Test Positions
    N_POSITIONS = 100
    test_positions = []
    for _ in tqdm(range(N_POSITIONS), desc="Generating Positions"):
        n_moves = random.randint(5, 20) # Create positions with some history
        board, player, move_seq = generate_random_state(n_moves)
        test_positions.append((board, player, move_seq))

    # 3. Run Benchmark
    for board, player, move_seq in tqdm(test_positions, desc="Running Benchmark"):
        # Get ground truth move
        correct_move = solved_bot.get_next_best_move(move_seq)
        if correct_move is None:
            continue # Skip if solver fails
        ground_truth.append(correct_move - 1) # Convert to 0-indexed

        # Get moves from other bots
        for name, bot in bots.items():
            action = bot.get_action(board, player)
            results[name].append(action)

    # 4. Analyze and Plot Results
    for name, bot_moves in results.items():
        if not bot_moves: continue
        # ensure ground_truth and bot_moves are the same length
        if len(ground_truth) != len(bot_moves):
            min_len = min(len(ground_truth), len(bot_moves))
            ground_truth = ground_truth[:min_len]
            bot_moves = bot_moves[:min_len]
            
        acc = accuracy_score(ground_truth, bot_moves)
        print(f"--- Results for {name} ---")
        print(f"Accuracy vs SolvedBot: {acc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(ground_truth, bot_moves, labels=list(range(7)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 8), yticklabels=range(1, 8))
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Move')
        plt.ylabel('True Move')
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{name}.png'))
        plt.close() # Close the figure to avoid displaying it in the notebook

if __name__ == '__main__':
    run_benchmark()
