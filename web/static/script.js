document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const newGameButton = document.getElementById('new-game');
    const toggleStarterButton = document.getElementById('toggle-starter');
    const statusDisplay = document.getElementById('status');
    const evalBarPlayer1 = document.getElementById('eval-bar-player1');
    const evalBarPlayer2 = document.getElementById('eval-bar-player2');

    let board = [];
    let gameEnded = false;
    let humanStarts = false; // Bot starts by default

    const updateEvalBar = (evaluation) => {
        // evaluation is from -1 to 1. P1's advantage.
        const player1Percent = (evaluation + 1) / 2 * 100;
        const player2Percent = 100 - player1Percent;

        evalBarPlayer1.style.height = `${player1Percent}%`;
        evalBarPlayer2.style.height = `${player2Percent}%`;
    };

    const createBoard = () => {
        gameBoard.innerHTML = '';
        board = Array(6).fill(null).map(() => Array(7).fill(0));
        for (let row = 0; row < 6; row++) {
            for (let col = 0; col < 7; col++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.classList.add('background-color');
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
        const animationPromises = changes.map(change => animateDrop(change.col, change.row, change.player));
        await Promise.all(animationPromises);
        updateBoard(newBoard);
    };

    const fetchEval = async () => {
        try {
            const evaluation_response = await fetch("/game/eval", {
                method: 'GET',
            });
            const evaluation = await evaluation_response.json();
            if (evaluation.eval !== undefined) {
                updateEvalBar(evaluation.eval);
            }
        } catch (error) {
            console.error('Error fetching evaluation:', error);
        }
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
                awaiting = false;
                return;
            }

            await findAndAnimateChanges(oldBoard, data.board);
            oldBoard = JSON.parse(JSON.stringify(data.board));
            
            await fetchEval();
            
            if (data.winner !== null && data.winner !== undefined && data.winner !== 0) {
                handleWinner(data.winner);
                awaiting = false;
                return;
            } else {
                setStatus("Bot turn");
            }
    

            const response2 = await fetch(`/game/moveB`, {
                method: 'POST',
            });
            const data2 = await response2.json();
            
            if (data2.error) {
                setStatus(data2.error);
                awaiting = false;
                return;
            }

            await findAndAnimateChanges(oldBoard, data2.board);
            oldBoard = JSON.parse(JSON.stringify(data2.board));

            await fetchEval();

            if (data2.winner !== null && data2.winner !== undefined) {
                handleWinner(data2.winner);
            } else {
                setStatus("Your turn");
            }

            awaiting = false;

        } catch (error) {
            console.error('Error making move:', error);
            setStatus('Error making move.');
            awaiting = false;
        }
    };

    const handleWinner = (winner) => {
        gameEnded = true;
        const botIsPlayer1 = !humanStarts;

        if (winner === 0) {
            setStatus("It's a draw!");
            updateEvalBar(0);
            return;
        }

        if ((winner === 1 && botIsPlayer1) || (winner === -1 && !botIsPlayer1)) {
            setStatus('Bot wins!');
        } else {
            setStatus('You win!');
        }
        updateEvalBar(winner);
    }

    const setStatus = (message) => {
        statusDisplay.textContent = message;
    };

    const startNewGame = async () => {
        gameEnded = false;
        awaiting = false;
        updateEvalBar(0); // Reset bar to 50/50

        const startPlayer = humanStarts ? 'human' : 'bot';
        setStatus(humanStarts ? "Your turn" : "Bot turn");

        try {
            const response = await fetch(`/game?start=${startPlayer}`, {
                method: 'POST',
            });
            const data = await response.json();
            const oldBoard = board;
            // await findAndAnimateChanges(oldBoard, data.board);
            updateBoard(data.board);
            await fetchEval();
            setStatus("Your turn");
        } catch (error) {
            console.error('Error starting new game:', error);
            setStatus('Error starting new game.');
        }
    };

    newGameButton.addEventListener('click', startNewGame);
    toggleStarterButton.addEventListener('click', () => {
        humanStarts = !humanStarts;
        toggleStarterButton.textContent = humanStarts ? 'You Start' : 'Bot Starts';
        startNewGame();
    });

    createBoard();
    startNewGame();
});

