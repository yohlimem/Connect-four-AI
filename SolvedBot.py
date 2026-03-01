import subprocess
import os
import time

class SolvedBot:
    def __init__(self, solver_path='./SolvedBotCPP/c4solver.exe'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.exe_full_path = os.path.normpath(os.path.join(current_dir, solver_path))

        if not os.path.exists(self.exe_full_path):
            raise FileNotFoundError(f"Solver not found: {self.exe_full_path}")

    def get_next_best_move(self, move_sequence: str) -> int:
        try:
            process = subprocess.Popen(
                [self.exe_full_path, '-a', '-b', f'{self.exe_full_path.strip("/c4solver.exe")}/7x6.book'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge errors into stdout so we see everything
                text=True,
                bufsize=1
            )
            stdout, _ = process.communicate(input=move_sequence + '\n', timeout=5)

            # Read result
            output = stdout.strip()
            
            if not output:
                return None

            parts = output.split()
            # If the solver outputs a warning about the book, the line might 
            # not start with the sequence. We filter for digit-based scores.
            scores = [int(s) for s in parts if s.lstrip('-').isdigit()]
            
            if len(scores) > 7: # If it echoed the moves, ignore the first element
                scores = scores[1:]

            if not scores: return None
            
            best_col = scores.index(max(scores)) + 1
            return best_col

        except Exception as e:
            print(f"Sigma, we had a glitch: {e}")
            return None

# --- Execution Test ---
if __name__ == "__main__":
    print("Initializing Sigma-Engine...")
    ai = SolvedBot()
    
    print("Engine ready. Calculating...")
    moves = "4"
    next_move = ai.get_next_best_move(moves)
    
    print(f"Result: {next_move}")
    def debug_solver(move_sequence, solver_path='./SolvedBotCPP/c4solver.exe'):
        # We use a context manager to ensure the process is cleaned up
        process = subprocess.Popen(
            [solver_path, '-a', '-b', f'{solver_path.strip('/c4solver.exe')}/7x6.book'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge errors into stdout so we see everything
            text=True,
            bufsize=1
        )

        try:
            print(f"--- Sigma Debug: Sending '{move_sequence}' to Solver ---")
            # communicate() sends the input and waits for the process to finish or timeout
            # This is MUCH safer than readline() for debugging.
            stdout, _ = process.communicate(input=move_sequence + '\n', timeout=2)
            
            print("--- Subprocess Output Start ---")
            print(stdout)
            print("--- Subprocess Output End ---")
            
        except subprocess.TimeoutExpired:
            print("!!! STALL DETECTED: The solver took too long to respond.")
            process.kill()
            # Grab whatever it managed to output before it stalled
            out, err = process.communicate()
            print(f"Partial output before kill: {out}")
        except Exception as e:
            print(f"An error occurred: {e}")
    debug_solver('4')