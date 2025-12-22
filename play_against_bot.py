from policy import Policy
import torch
from connect4 import Connect4Env
from utils import preprocess_board



def play_against_bot(env: Connect4Env, bot: Policy, player_start:bool = True):
    '''
    player start: whether the player starts or the bot
    '''
    state, _ = env.reset()
    done = False
    info = None

    def get_action(board, current_player):
        ai_input = torch.from_numpy(preprocess_board(board, current_player)).float().unsqueeze(0).to("cuda")
        logits = ppo_policy.forward(ai_input)[0]
        mask = torch.tensor([0.0 if env.board[0][c] == 0 else -1e9 for c in range(env.cols)]).to("cuda")
        return torch.argmax(logits + mask).item()
    
    env.render()
    while not done:
        action = 8
        if env.current_player == (1 if not player_start else -1):
            action = get_action(env.board, env.current_player)
            print(f"bot played column: {action+1}")
        else:
            try:
                action = int(input("where would you like to place your pieces? (1-7)")) - 1
                while env.board[0][action] != 0 or action < 0 or action > 6:
                    print("Invalid move")
                    action = int(input("where would you like to place your pieces? (1-7)")) - 1
            except ValueError:
                print("Invalid move")
                continue
        
            

            
        obs, _, done, _, info = env.step(action)
        env.render()

    if info["winner"] == (1 if player_start else -1):
        print("Player won")
    elif info["winner"] == 0:
        print("Draw")
    elif info is None:
        print("Something went wrong!")
    else:
        print("Bot won")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env =  Connect4Env()

    ppo_policy = Policy(env.action_space.n, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256]).to(device)
    ppo_policy.load_from_file("This bot is super good large CNN.pth", device=device)
    play_against_bot(env, player_start=True, bot=ppo_policy)

        