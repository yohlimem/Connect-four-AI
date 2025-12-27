from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from torch.distributions import Categorical
from typing import Literal

from connect4 import Connect4Env
from policy import Policy
from utils import preprocess_board

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

game = Connect4Env()
bot: Policy | None= None
device = "cuda" if torch.cuda.is_available() else "cpu"
bot_player_id = 1

@app.on_event("startup")
def load_bot():
    global bot
    bot = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256]).to(device)

    # bot.load_from_file("This bot is super good large CNN.pth", device=device)
    bot.load_from_file(".\\Saves\\bots\\connect4_parallel_iter_11000.pth", device=device)
    bot.to(device)
    bot.eval()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/game")
async def start_game(start: Literal['bot', 'human'] = 'bot'):
    global game, bot_player_id
    game.reset()
    if start == 'bot':
        bot_player_id = 1
        # Bot's turn
        action = get_bot_move()
        game.step(action)
    else: # human starts
        bot_player_id = -1
    
    return {"board": game.board.tolist()}

def get_bot_move():
    global game, bot, bot_player_id
    if bot is None:
        raise Exception("Bot not loaded")

    # Preprocess board and convert to tensor
    preprocessed_board = preprocess_board(game.board, current_player=bot_player_id)
    board_tensor = torch.tensor(preprocessed_board, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = bot(board_tensor)[0]

    # Mask invalid actions (full columns)
    valid_actions_mask = torch.tensor([game.board[0][col] == 0 for col in range(7)], dtype=torch.bool).to(device)
    logits[~valid_actions_mask] = -float('inf')

    action = logits.argmax().item()

    return action

@app.post("/game/moveP")
async def make_move_player(column: int):
    global game, bot_player_id
    if game is None or game.is_done():
        return {"error": "Game is not active."}
    
    if game.current_player == bot_player_id:
        return {"error": "Not your turn.", "board": game.board.tolist()}

    if game.board[0][column] != 0:
        return {"error": "Invalid move: column is full.", "board": game.board.tolist()}
        
    _, _, done, _, info = game.step(column)

    return {"board": game.board.tolist(), "winner": info.get("winner") if done else None}

@app.post("/game/moveB")
async def make_move_bot():
    global game, bot_player_id
    if game is None or game.is_done():
        return {"error": "Game is not active."}

    if game.current_player != bot_player_id:
        return {"error": "Not bot's turn.", "board": game.board.tolist()}

    # Bot's turn
    action = get_bot_move()
    _, _, done, _, info = game.step(action)

    return {"board": game.board.tolist(), "winner": info.get("winner") if done else None}
    
@app.get("/game/eval")
async def eval_position():
    global game, bot_player_id
    if game is None or game.is_done():
        return {"error": "Game is not active."}

    # Get evaluation from the bot's perspective.
    evaluation = bot.value_function(torch.tensor(preprocess_board(game.board, current_player=bot_player_id)).unsqueeze(0).to(device))
    
    # Eval bar is always from Player 1's perspective.
    # If bot is Player 2, its evaluation is from P2's perspective. We need to flip it for P1's perspective.
    if bot_player_id == -1:
        evaluation *= -1
        
    return {"eval": evaluation.item()}

@app.get("/game/state")
async def get_state():
    if game is None:
        return {"error": "Game not started"}
    return {"board": game.board.tolist()}
