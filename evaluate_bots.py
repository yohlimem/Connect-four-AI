import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 

import random
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from connect4 import Connect4Env
from policy import Policy
from SolvedBot import SolvedBot

# --- CONFIGURATION ---
BOT_FOLDER = "Saves/CNN"
NUM_POSITIONS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. GENERATE EVALUATION SET (IN-MEMORY) ---

def generate_random_position():
    """Generates a random board position by making a random number of moves."""
    env = Connect4Env()
    move_sequence = []
    num_moves = random.randint(5, 30)  # Make the positions non-trivial
    for _ in range(num_moves):
        if env.is_done():
            break
        
        possible_moves = [c for c in range(env.cols) if env.board[0, c] == 0]
        if not possible_moves:
            break
            
        move = random.choice(possible_moves)
        env.step(move)
        move_sequence.append(str(move + 1)) # SolvedBot uses 1-based indexing for columns

    return env, "".join(move_sequence)

def prepare_evaluation_set(num_positions):
    """
    Generates a set of random positions and their corresponding
    best moves from the solved bot, holding them in memory.
    """
    print(f"Generating {num_positions} evaluation positions...")
    solved_bot = SolvedBot()
    board_list, player_list, solved_move_list = [], [], []

    while len(board_list) < num_positions:
        env, move_sequence = generate_random_position()

        if env.is_done():
            continue

        try:
            solved_move_col = solved_bot.get_next_best_move(move_sequence)
            if solved_move_col is None:
                continue
            
            board_list.append(env.board)
            player_list.append(env.current_player)
            solved_move_list.append(solved_move_col - 1) # Convert to 0-indexed
            
            print(f"Generated {len(board_list)}/{num_positions}", end="\r")

        except Exception as e:
            print(f"Could not get solved move: {e}")
            continue
    
    print(f"\nFinished generating {len(board_list)} positions.")
    return np.array(board_list), np.array(player_list), np.array(solved_move_list)


# --- 2. LOAD BOTS ---
def get_bot_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pth")]

def get_iteration_from_filename(filename):
    match = re.search(r'iter_(\d+)', filename)
    return int(match.group(1)) if match else 0


# --- 3. BATCH EVALUATION ---
def evaluate_bots():
    # Prepare the evaluation set in memory
    boards_np, players_np, solved_moves_np = prepare_evaluation_set(NUM_POSITIONS)
    num_eval_positions = len(boards_np)

    if num_eval_positions == 0:
        print("No evaluation positions were generated.")
        return [], []

    bot_files = get_bot_files(BOT_FOLDER)
    accuracies = {}
    iterations = {}

    # Load all models
    models = {}
    for bot_file in bot_files:
        iteration = get_iteration_from_filename(bot_file)
        if iteration == 0:
            continue
        try:
            policy = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256]).to(DEVICE)
            policy.load_state_dict(torch.load(bot_file, map_location=DEVICE))
            policy.eval()
            models[iteration] = policy
            iterations[iteration] = iteration
        except Exception as e:
            print(f"Could not load model {bot_file}: {e}")

    # Batch process all boards for each model
    for iteration, model in models.items():
        print(f"Evaluating model from iteration {iteration}...")

        # Preprocess all boards at once.
        # The board is flipped so the model always sees itself as player '1'.
        relative_boards = boards_np * players_np.reshape(-1, 1, 1)
        
        my_pieces = (relative_boards == 1).astype(np.float32)
        enemy_pieces = (relative_boards == -1).astype(np.float32)
        
        # input_tensor shape: (N, 2, 6, 7)
        input_tensor = torch.from_numpy(np.stack([my_pieces, enemy_pieces], axis=1)).float().to(DEVICE)
        
        with torch.no_grad():
            policy_outputs = model(input_tensor) # (N, 7)

        # Mask illegal moves for the entire batch
        top_row = boards_np[:, 0, :] # (N, 7)
        legal_moves_mask = torch.from_numpy(top_row == 0).to(DEVICE) # (N, 7)
        
        policy_outputs[~legal_moves_mask] = -torch.inf

        # Get model's best move for each position
        bot_moves_np = torch.argmax(policy_outputs, dim=1).cpu().numpy() # (N,)

        # Calculate accuracy
        correct_predictions = (bot_moves_np == solved_moves_np).sum()
        accuracies[iteration] = correct_predictions / num_eval_positions

    print("\nEvaluation complete.")
    return sorted(iterations.values()), [accuracies.get(i, 0) for i in sorted(iterations.values())]


# --- 4. PLOTTING ---
def create_graph(iterations, accuracies):
    if not iterations:
        print("No models were evaluated. Skipping graph creation.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o')
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy vs SolvedBot")
    plt.title("Connect4 AI Performance")
    plt.grid(True)
    plt.savefig("evaluation_graph.png")
    print("Graph saved to evaluation_graph.png")


# --- MAIN ---
if __name__ == "__main__":
    iterations, accuracies = evaluate_bots()
    create_graph(iterations, accuracies)