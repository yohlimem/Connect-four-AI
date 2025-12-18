import numpy as np
import torch

from connect4 import Connect4Env




def test_sigma_agent(env_class, policy = None, policy_2 = None, games=100): # (made with gemini)
    env = env_class()
    ai_wins = 0
    
    print(f"Running {games} games against Random Bot...")
    
    for game in range(games):
        obs, _ = env.reset()
        # AI plays as Player 1 (Blue)
        done = False
        
        while not done:
            if env.current_player == 1: # AI Turn
                # AI Logic
                if policy is not None:
                    ai_input = torch.from_numpy(obs.flatten() * 1).float().unsqueeze(0)
                    with torch.no_grad():
                        logits = policy(ai_input)[0]
                        mask = torch.tensor([0.0 if env.board[0][c] == 0 else -1e9 for c in range(env.cols)])
                        action = torch.argmax(logits + mask).item()
                else:
                    print("please give policy to check")
                    return
            else: # Random Turn
                if policy_2 is not None:
                    ai_input = torch.from_numpy(obs.flatten() * 1).float().unsqueeze(0)
                    with torch.no_grad():
                        logits = policy_2(ai_input)[0]
                        mask = torch.tensor([0.0 if env.board[0][c] == 0 else -1e9 for c in range(env.cols)])
                        action = torch.argmax(logits + mask).item()
                    
                else:
                    # Random Logic
                    legal = [c for c in range(env.cols) if env.board[0][c] == 0]
                    action = np.random.choice(legal)
                
            obs, _, done, _, info = env.step(action)
            
        if info['winner'] == 1:
            ai_wins += 1
            
    print(f"Sigma AI Win Rate vs Random: {ai_wins}/{games} ({ai_wins/games*100}%)")


if __name__ == '__main__':
    env = Connect4Env()
    from policy import Policy
    from state_value import StateValue
    ppo_policy = Policy(StateValue(env.observation_space.shape[0] * env.observation_space.shape[1], 3, 256), env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n, 3, 256, ent_coef=0.03) # pyright: ignore[reportOptionalSubscript]
    ppo_policy_2 = Policy(StateValue(env.observation_space.shape[0] * env.observation_space.shape[1], 3, 256), env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n, 3, 256, ent_coef=0.03) # pyright: ignore[reportOptionalSubscript]
    ppo_policy.load_from_file("connect4_policy_iter_600000.pth")
    test_sigma_agent(Connect4Env, policy=ppo_policy, policy_2=ppo_policy_2)
    

    