from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from typing import Literal

from database import init_db, save_game
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
move_history = []
current_move_index = -1

@app.on_event("startup")
def load_bot_and_init_db():
    """Load the bot and initialize the database."""
    global bot
    init_db()
    bot = Policy(7, input_channels=2, board_height=6, board_width=7, ent_coef=0.03, conv_layers_channels=[128, 64, 32], fc_layer_sizes=[512, 512, 256]).to(device)

    try:
        # bot.load_from_file("This bot is super good large CNN.pth", device=device)
        bot.load_from_file("./best_currently_even_more.pth", device=device)
        # bot.load_from_file(".\\Saves\\bots\\beat_alpha_beta_bot.pth", device=device)
        bot.to(device)
        bot.eval()
    except FileNotFoundError:
        print("Bot model not found. The bot will not be used.")
        bot = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/game")
async def start_game(start: Literal['bot', 'human'] = 'bot'):
    global game, bot_player_id, move_history, current_move_index
    game.reset()
    move_history = []
    current_move_index = -1
    
    if start == 'bot':
        bot_player_id = 1
        # Bot's turn
        action = get_bot_move()
        game.step(action)
        move_history.append(action)
        current_move_index += 1
    else: # human starts
        bot_player_id = -1
    
    return {"board": game.board.tolist(), "move_history": move_history, "current_move_index": current_move_index}

def get_bot_move():
    global game, bot, bot_player_id
    if bot is None:
        raise Exception("Bot not loaded")

    # Preprocess board and convert to tensor
    preprocessed_board = preprocess_board(game.board, current_player=game.current_player)
    board_tensor = torch.tensor(preprocessed_board, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = bot(board_tensor)[0]

    # Mask invalid actions (full columns)
    valid_actions_mask = torch.tensor([game.board[0][col] == 0 for col in range(7)], dtype=torch.bool).to(device)
    logits[~valid_actions_mask] = -float('inf')

    action = logits.argmax().item()

    return action

def handle_move(column: int):
    global game, move_history, current_move_index, bot_player_id

    # If we are not at the end of the history, we are branching.
    if current_move_index < len(move_history) - 1:
        game.reset()
        for i in range(current_move_index + 1):
            game.step(move_history[i])
        
        move_history = move_history[:current_move_index + 1]


    _, _, done, _, info = game.step(column)
    move_history.append(column)
    current_move_index += 1
    
    winner = info.get("winner")
    if done and winner is not None:
        save_game(moves=move_history, winner=winner, bot_player_id=bot_player_id)
    
    return {"board": game.board.tolist(), "winner": winner if done else None, "move_history": move_history, "current_move_index": current_move_index}


@app.post("/game/moveP")
async def make_move_player(column: int):
    global game, bot_player_id
    # Allow moves even if game is done to allow for branching
    if game.current_player == bot_player_id and not game.is_done():
        return {"error": "Not your turn.", "board": game.board.tolist()}

    if game.board[0][column] != 0 and not (current_move_index < len(move_history) -1):
        return {"error": "Invalid move: column is full.", "board": game.board.tolist()}
        
    return handle_move(column)

@app.post("/game/moveB")
async def make_move_bot():
    global game, bot_player_id
    # Allow moves even if game is done to allow for branching
    if game.current_player != bot_player_id and not game.is_done():
        return {"error": "Not bot's turn.", "board": game.board.tolist()}

    # Bot's turn
    action = get_bot_move()
    return handle_move(action)
    
@app.post("/game/navigate")
async def navigate_history(direction: Literal['back', 'forward']):
    global current_move_index, move_history, game
    print("Navigating")

    if direction == 'back' and current_move_index > -1:
        current_move_index -= 1
    elif direction == 'forward' and current_move_index < len(move_history) - 1:
        current_move_index += 1
    
    game.reset()
    winner = None
    for i in range(current_move_index + 1):
        _, _, done, _, info = game.step(move_history[i])
        if done:
            winner = info.get("winner")
        
    return {"board": game.board.tolist(), "move_history": move_history, "current_move_index": current_move_index, "winner": winner}

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
    return {"board": game.board.tolist(), "move_history": move_history, "current_move_index": current_move_index}
