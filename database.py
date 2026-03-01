import sqlite3
import json
from datetime import datetime

DB_NAME = "connect4_games.db"

def init_db():
    """Initializes the database and creates the 'games' table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            moves TEXT NOT NULL,
            winner INTEGER NOT NULL,
            bot_player_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_game(moves: list, winner: int, bot_player_id: int):
    """Saves a completed game to the database.

    Args:
        moves (list): A list of column indices representing the moves made.
        winner (int): The winner of the game (1, -1, or 0 for a draw).
        bot_player_id (int): The player ID of the bot (1 or -1).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    moves_json = json.dumps(moves)
    timestamp = datetime.utcnow()
    
    cursor.execute("""
        INSERT INTO games (moves, winner, bot_player_id, timestamp)
        VALUES (?, ?, ?, ?)
    """, (moves_json, winner, bot_player_id, timestamp))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Example of how to use the functions
    print("Initializing the database...")
    init_db()
    print("Database initialized.")

    # Example game data
    example_moves = [3, 4, 3, 2, 3, 1, 3]
    example_winner = 1  # Player 1 wins
    example_bot_player = -1

    print("Saving an example game...")
    save_game(example_moves, example_winner, example_bot_player)
    print("Example game saved.")

    # You can use a tool like DB Browser for SQLite to view the connect4_games.db file
    # Or query it with python's sqlite3 library.
