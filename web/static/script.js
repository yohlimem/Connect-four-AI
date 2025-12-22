document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const newGameButton = document.getElementById('new-game');
    const statusDisplay = document.getElementById('status');

    let board = [];
    let gameEnded = false;

    const createBoard = () => {
        gameBoard.innerHTML = '';
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 7; col++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.addEventListener('click', () => handleCellClick(col));
                gameBoard.appendChild(cell);
            }
        }
    };

    const updateBoard = (newBoard) => {
        board = newBoard;
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 7; col++) {
                const cell = gameBoard.querySelector(`[data-row='${row}'][data-col='${col}']`);
                cell.classList.remove('player1', 'player2');
                // Bot is player 1 (red), Human is player -1 (yellow)
                if (board[row][col] === 1) {
                    cell.classList.add('player1');
                } else if (board[row][col] === -1) {
                    cell.classList.add('player2');
                }
            }
        }
    };

    const handleCellClick = async (col) => {
        if (gameEnded) {
            setStatus("Game over. Please start a new game.");
            return;
        }

        try {
            const response = await fetch(`/game/move?column=${col}`, {
                method: 'POST',
            });
            const data = await response.json();

            if (data.error) {
                setStatus(data.error);
                return;
            }

            updateBoard(data.board);

            if (data.winner !== null && data.winner !== undefined) {
                handleWinner(data.winner);
            } else {
                setStatus("Your turn");
            }

        } catch (error) {
            console.error('Error making move:', error);
            setStatus('Error making move.');
        }
    };

    const handleWinner = (winner) => {
        gameEnded = true;
        if (winner === 1) {
            setStatus('Bot wins!');
        } else if (winner === -1) {
            setStatus('You win!');
        } else if (winner === 0) {
            setStatus("It's a draw!");
        }
    }

    const setStatus = (message) => {
        statusDisplay.textContent = message;
    };

    const startNewGame = async () => {
        gameEnded = false;
        try {
            const response = await fetch(`/game`, {
                method: 'POST',
            });
            const data = await response.json();
            console.log(data)
            updateBoard(data.board);
            setStatus("Your turn");
        } catch (error) {
            console.error('Error starting new game:', error);
            setStatus('Error starting new game.');
        }
    };

    newGameButton.addEventListener('click', startNewGame);

    createBoard();
    startNewGame();
});
