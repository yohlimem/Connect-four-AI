document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const newGameButton = document.getElementById('new-game');
    const statusDisplay = document.getElementById('status');

    let board = [];
    let gameEnded = false;

    const createBoard = () => {
        gameBoard.innerHTML = '';
        board = Array(6).fill(null).map(() => Array(7).fill(0));
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

    const animateDrop = (col, row, player) => {
        return new Promise(resolve => {
            const token = document.createElement('div');
            token.classList.add('token', player === 1 ? 'player1' : 'player2');
            token.style.left = `${col * 85 + 5}px`; // 80px cell + 5px gap

            gameBoard.appendChild(token);

            const endTop = row * 85 + 5; // 80px cell + 5px gap
            
            setTimeout(() => {
                token.style.transform = `translateY(${endTop}px)`;
            }, 20);

            token.addEventListener('transitionend', () => {
                token.remove();
                resolve();
            });
        });
    };

    const findAndAnimateChanges = async (oldBoard, newBoard) => {
        const changes = [];
        for (let r = 0; r < 6; r++) {
            for (let c = 0; c < 7; c++) {
                if (oldBoard[r][c] !== newBoard[r][c]) {
                    changes.push({ row: r, col: c, player: newBoard[r][c] });
                }
            }
        }
        console.log(changes)
        const animationPromises = changes.map(change => animateDrop(change.col, change.row, change.player));
        await Promise.all(animationPromises);
        updateBoard(newBoard);
    };


    let awaiting = false;
    const handleCellClick = async (col) => {
        if (awaiting){
            return;
        }
        if (gameEnded) {
            setStatus("Game over. Please start a new game.");
            return;
        }

        let oldBoard = JSON.parse(JSON.stringify(board));

        try {
            awaiting = true;
            const response = await fetch(`/game/moveP?column=${col}`, {
                method: 'POST',
            });
            const data = await response.json();

            if (data.error) {
                setStatus(data.error);
                return;
            }
            
            await findAndAnimateChanges(oldBoard, data.board);
            oldBoard = JSON.parse(JSON.stringify(data.board));
            
            if (data.winner !== null && data.winner !== undefined) {
                handleWinner(data.winner);
            } else {
                setStatus("Bot turn");
            }
    

            const response2 = await fetch(`/game/moveB`, {
                method: 'POST',
            });
            const data2 = await response2.json();
            
            if (data2.error) {
                setStatus(data2.error);
                return;
            }

            await findAndAnimateChanges(oldBoard, data2.board);
            oldBoard = JSON.parse(JSON.stringify(data2.board));

            if (data2.winner !== null && data2.winner !== undefined) {
                handleWinner(data2.winner);
            } else {
                setStatus("Your turn");
            }

            awaiting = false;



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
        awaiting = false;
        setStatus("Bot turn");

        const oldBoard = JSON.parse(JSON.stringify(board));
        try {
            const response = await fetch(`/game`, {
                method: 'POST',
            });
            const data = await response.json();
            console.log(data)
            updateBoard(data.board)
            // await findAndAnimateChanges(oldBoard, data.board);
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

