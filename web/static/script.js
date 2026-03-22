document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const newGameButton = document.getElementById('new-game');
    const toggleStarterButton = document.getElementById('toggle-starter');
    const backMoveButton = document.getElementById('back-move');
    const forwardMoveButton = document.getElementById('forward-move');
    const statusDisplay = document.getElementById('status');
    const evalBarPlayer1 = document.getElementById('eval-bar-player1');
    const evalBarPlayer2 = document.getElementById('eval-bar-player2');
    const moveList = document.getElementById('move-list');

    let board = [];
    let gameEnded = false;
    let botStarts = true; // Bot starts by default
    let playerColors = {1: 'White', '-1': 'Black'};

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

    const updateBoard = (data) => {
        const newBoard = data.board;
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
        if (data.move_history !== undefined) {
            updateMoveHistory(data.move_history, data.current_move_index, botStarts);
        }
        if (data.winner) {
            handleWinner(data.winner);
        } else {
            gameEnded = false;
        }
    };

    const updateMoveHistory = (moves, currentIndex, botStarts) => {
        moveList.innerHTML = '';
        moves.forEach((move, index) => {
            const li = document.createElement('li');
            const player = (index % 2 === 0) ? (botStarts ? 'Bot' : 'You') : (botStarts ? 'You' : 'Bot');
            const playerColor = (index % 2 === 0) ? (botStarts ? playerColors[1] : playerColors[-1]) : (botStarts ? playerColors[-1] : playerColors[1]);
            li.textContent = `${player} (${playerColor}): Column ${move + 1}`;
            
            if (index === currentIndex) {
                li.classList.add('active');
            }

            li.addEventListener('click', async () => {
                if (awaiting) return;
                const newMoveIndex = index;
                let diff = newMoveIndex - window.currentMoveIndex;
                const direction = diff > 0 ? 'forward' : 'back';
                diff = Math.abs(diff);

                for (let i = 0; i < diff; i++) {
                   await navigateHistory(direction, false)
                }
                await fetchEval();

            });
            moveList.appendChild(li);
        });
        window.currentMoveIndex = currentIndex;
        moveList.scrollTop = moveList.scrollHeight;
    }

    const animateDrop = (col, row, player) => {
        return new Promise(resolve => {
            const token = document.createElement('div');
            token.classList.add('token', player === 1 ? 'player1' : 'player2');
            token.style.left = `${col * 85 + 5}px`;

            gameBoard.appendChild(token);

            const endTop = row * 85 + 5;
            
            setTimeout(() => {
                token.style.transform = `translateY(${endTop}px)`;
            }, 20);

            token.addEventListener('transitionend', () => {
                token.remove();
                resolve();
            });
        });
    };

    const findAndAnimateChanges = async (oldBoard, data) => {
        const newBoard = data.board;
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
        updateBoard(data);
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

    const handleServerResponse = async (response) => {
        const data = await response.json();
        if (data.error) {
            setStatus(data.error);
            return {gameOver: false, error: true};
        }

        const oldBoard = JSON.parse(JSON.stringify(board));
        await findAndAnimateChanges(oldBoard, data);
        await fetchEval();

        if (data.winner !== null && data.winner !== undefined && data.winner !== 0) {
            handleWinner(data.winner);
            return {gameOver: true, error: false};
        }
        return {gameOver: false, error: false};
    };

    const handleCellClick = async (col) => {
        if (awaiting || gameEnded){
            return;
        }
        
        awaiting = true;
        
        try {
            // Player's move
            setStatus("Your turn");
            const playerResponse = await fetch(`/game/moveP?column=${col}`, { method: 'POST' });
            let { gameOver, error } = await handleServerResponse(playerResponse);
            if (gameOver || error) {
                awaiting = false;
                return;
            }
            
            // Bot's move
            setStatus("Bot's turn");
            const botResponse = await fetch(`/game/moveB`, { method: 'POST' });
            ({ gameOver, error } = await handleServerResponse(botResponse));
            if (gameOver || error) {
                awaiting = false;
                return;
            }

            setStatus("Your turn");

        } catch (error) {
            console.error('Error making move:', error);
            setStatus('Error making move.');
        } finally {
            awaiting = false;
        }
    };

    const handleWinner = (winner) => {
        gameEnded = true;

        if (winner === 0) {
            setStatus("It's a draw!");
            updateEvalBar(0);
            return;
        }

        if ((winner === 1 && !botStarts) || (winner === -1 && botStarts)) {
            setStatus('You win!');
        } else {
            setStatus('Bot wins!');
        }
        updateEvalBar(winner);
    }

    const setStatus = (message) => {
        statusDisplay.textContent = message;
    };

    const startNewGame = async () => {
        gameEnded = false;
        awaiting = false;
        updateEvalBar(0);

        const startPlayer = botStarts ? 'bot' : 'human';
        
        try {
            const response = await fetch(`/game?start=${startPlayer}`, {
                method: 'POST',
            });
            const data = await response.json();
            updateBoard(data);
            await fetchEval();
            
            if (botStarts) {
                setStatus("Your turn");
            } else {
                setStatus("Your turn");
            }

        } catch (error) {
            console.error('Error starting new game:', error);
            setStatus('Error starting new game.');
        }
    };

    const navigateHistory = async (direction, do_animation = true) => {
        if (awaiting) return;
        awaiting = true;
        try {
            const response = await fetch(`/game/navigate?direction=${direction}`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.error) {
                setStatus(data.error);
            }
            else {
                const oldBoard = JSON.parse(JSON.stringify(board));
                // if (do_animation){
                //     await findAndAnimateChanges(oldBoard, data);
                // } else {
                updateBoard(data);
                // }
                await fetchEval();
            }
        } catch (error) {
            console.error('Error navigating history:', error);
            setStatus('Error navigating history.');
        }
        awaiting = false;
    };
    console.log("amongus killyouse")

    newGameButton.addEventListener('click', startNewGame);
    toggleStarterButton.addEventListener('click', () => {
        botStarts = !botStarts;
        toggleStarterButton.textContent = botStarts ? 'Bot Starts' : 'You Start';
        startNewGame();
    });
    backMoveButton.addEventListener('click', () => navigateHistory('back', true));
    forwardMoveButton.addEventListener('click', () => navigateHistory('forward', true));
    createBoard();
    startNewGame();
});
