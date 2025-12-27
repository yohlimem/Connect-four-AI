import glob
import torch
import numpy as np
from connect4 import Connect4Env
from policy import Policy
from utils import preprocess_board
from AlphaBetaBot import AlphaBetaBot
from datetime import datetime as dt
class BotPlayer:
    """
    A wrapper class for different bot types to provide a unified interface.
    """
    def __init__(self, bot_instance, name, bot_type='nn'):
        self.bot_instance = bot_instance
        self.name = name
        self.bot_type = bot_type

    def get_move(self, board, player_id):
        if self.bot_type == 'nn':
            # For neural network bots (Policy)
            preprocessed_board = preprocess_board(board, player_id)
            with torch.no_grad():
                logits = self.bot_instance(torch.tensor(preprocessed_board, dtype=torch.float32).unsqueeze(0))[0]
                # Mask invalid actions (full columns)
                valid_actions_mask = torch.tensor([board[0][col] == 0 for col in range(7)], dtype=torch.bool)
                logits[~valid_actions_mask] = -float('inf')
                # Get the action with the highest logit
                action = logits.argmax().item()
            return action
        elif self.bot_type == 'ab':
            # For AlphaBeta bots
            return self.bot_instance.get_action(board, player_id)
        else:
            raise ValueError(f"Unknown bot type: {self.bot_type}")



def play_game(player1: BotPlayer, player2: BotPlayer):
    """
    Plays a single game of Connect4 between two bots.
    """
    game = Connect4Env()
    players = {1: player1, -1: player2}
    
    while not game.is_done():
        current_player_id = game.current_player
        current_player = players[current_player_id]
        
        action = current_player.get_move(game.board, current_player_id)
        
        

        _,_,_,_,info = game.step(action)
        
    winner = info.get('winner')
    if winner:
        winner_name = players[winner].name
        print(f"Game over: {winner_name} wins!")
    else:
        print("Game over: It's a draw!")
        
    return winner, "Game finished"

def run_tournament(bots):
    """
    Runs a round-robin tournament for a list of bots.
    Each bot plays every other bot as Player 1.
    """
    bot_names = [bot.name for bot in bots]
    results = {name: {other_name: '' for other_name in bot_names} for name in bot_names}
    scores = {bot.name: {'wins': 0, 'draws': 0, 'losses': 0} for bot in bots}


    for i in range(len(bots)):
        for j in range(len(bots)):
            if i == j:
                results[bots[i].name][bots[j].name] = '---'
                continue

            bot1 = bots[i]  # Plays as P1
            bot2 = bots[j]  # Plays as P2

            print(f"\n--- Match: {bot1.name} (P1) vs {bot2.name} (P2) ---")
            winner, reason = play_game(bot1, bot2)

            if winner == 1:  # bot1 (P1) wins
                results[bot1.name][bot2.name] = 'Win'
                scores[bot1.name]['wins'] += 1
                scores[bot2.name]['losses'] += 1

            elif winner == -1:  # bot2 (P2) wins
                results[bot1.name][bot2.name] = 'Loss'
                scores[bot1.name]['losses'] += 1
                scores[bot2.name]['wins'] += 1

            else:  # Draw
                results[bot1.name][bot2.name] = 'Draw'
                scores[bot1.name]['draws'] += 1
                scores[bot2.name]['draws'] += 1


    return results, scores

def display_results_matrix(results, bots):
    """
    Displays the tournament results in a matrix format using Matplotlib.
    Rows represent Player 1, Columns represent Player 2.
    Saves the output to 'tournament_results.png'.
    """
    bot_names = [bot.name for bot in bots]
    short_names = [name[:15] for name in bot_names] # Truncate for display

    # Prepare data for the table
    cell_text = []
    cell_colours = []
    color_map = {'Win': 'lightgreen', 'Loss': 'lightcoral', 'Draw': 'lightyellow', '---': 'lightgrey'}

    for p1_name in bot_names:
        row_text = []
        row_colours = []
        for p2_name in bot_names:
            result = results[p1_name][p2_name]
            row_text.append(result)
            row_colours.append(color_map.get(result, 'white'))
        cell_text.append(row_text)
        cell_colours.append(row_colours)

    fig, ax = plt.subplots(figsize=(max(10, len(short_names)), max(5, len(short_names) * 0.5)))
    ax.axis('tight')
    ax.axis('off')

    table = plt.table(cellText=cell_text,
                      cellColours=cell_colours,
                      rowLabels=short_names,
                      colLabels=short_names,
                      loc='center',
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    ax.set_title('Tournament Results (Player 1 vs Player 2)', fontsize=16, pad=20)

    # Save the figure
    output_filename = f"./Saves/leaderboards/tournament_results_{dt.now().strftime('%d-%m-%Y_%H-%M-%S')}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"\nResults matrix saved to {output_filename}")
    
    # Optionally, show the plot
    # plt.show()
def display_leaderboard(scores):
    """
    Calculates total points, displays a sorted leaderboard, and saves it to a file.
    """
    leaderboard = []
    for bot_name, result in scores.items():
        points = result['wins'] * 3 + result['draws'] * 1
        leaderboard.append((bot_name, result['wins'], result['draws'], result['losses'], points))

    # Sort by points descending, then by wins
    leaderboard.sort(key=lambda x: (x[4], x[1]), reverse=True)

    # --- Print to console ---
    print("\n\n--- Leaderboard ---")
    header = f"{'Rank':<5} {'Bot':<40} {'Wins':<5} {'Draws':<6} {'Losses':<7} {'Points':<6}"
    print(header)
    print("-" * 75)
    
    leaderboard_str_console = []
    for i, (name, wins, draws, losses, points) in enumerate(leaderboard):
        line = f"{i+1:<5} {name:<40} {wins:<5} {draws:<6} {losses:<7} {points:<6}"
        leaderboard_str_console.append(line)
    
    print('\n'.join(leaderboard_str_console))

    # --- Save to file ---
    try:
        with open("Saves/leaderboards/leaderboard_history.txt", "a") as f:
            f.write(f"\n\n--- Leaderboard generated at: {dt.now().strftime('%d-%m-%Y %H:%M:%S')} ---\n")
            f.write(header + "\n")
            f.write("-" * 75 + "\n")
            for line in leaderboard_str_console:
                f.write(line + "\n")
        print("\nLeaderboard saved to leaderboard_history.txt")
    except Exception as e:
        print(f"\nError saving leaderboard: {e}")
if __name__ == '__main__':
    # Add matplotlib to imports
    import matplotlib.pyplot as plt

    # 1. Find and load all bots
    bots = []

    # Load AlphaBeta bot
    ab_bot = AlphaBetaBot(depth=7) # Using a reasonable depth
    bots.append(BotPlayer(ab_bot, "AlphaBetaBot (D7)", bot_type='ab'))

    policy1 = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256])
    policy1.load_from_file("This bot is super good large CNN.pth")
    bots.append(BotPlayer(policy1, "old best bot", bot_type='nn'))

    # Load all saved neural network bots
    model_paths = glob.glob('*.pth')
    model_paths.extend(glob.glob('Saves/bots/*.pth')) # Also check in saves folder
    model_paths.extend(glob.glob('Saves/SavedWorkBots/*.pth')) # Also check in saves folder

    for model_path in model_paths:
        try:
            policy = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256])
            policy.load_from_file(model_path)
            policy.eval() # Set to evaluation mode
            bot_name = model_path.replace('\\', '/').split('_')[-1]
            bots.append(BotPlayer(policy, bot_name, bot_type='nn'))
            print(f"Loaded bot: {bot_name}")
        except Exception as e:
            print(f"Could not load bot from {model_path}. Error: {e}")

    if len(bots) < 2:
        print("Not enough bots to run a tournament. Need at least 2.")
    else:
        # 2. Run the tournament
        final_results, final_scores = run_tournament(bots)
        display_leaderboard(final_scores)
        print(final_scores)
        # 3. Display the results
        display_results_matrix(final_results, bots)
