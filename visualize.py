import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
from PIL import Image, ImageDraw
import torch
import numpy as np

# Colors
BLUE = (29, 95, 219)
RED = (219, 29, 29)
YELLOW = (235, 222, 18)
WHITE = (255, 255, 255)

def tensor_to_board_image(tensor: torch.Tensor, cell_size=100):
    """
    Converts a (2, 6, 7) board tensor into a PIL image of the Connect Four board.

    Args:
        tensor (torch.Tensor): A tensor of shape (2, 6, 7) representing the board state.
                               Channel 0: Player 1's pieces (1s)
                               Channel 1: Player 2's pieces (1s)
        cell_size (int): The size of each cell in pixels.

    Returns:
        PIL.Image.Image: An image of the board.
    """
    if tensor.shape != (2, 6, 7):
        raise ValueError("Input tensor must have shape (2, 6, 7)")

    board_height, board_width = tensor.shape[1], tensor.shape[2]
    
    # Convert tensor to a numpy array board with 1, -1, 0
    my_pieces = tensor[0].cpu().numpy()
    enemy_pieces = tensor[1].cpu().numpy()
    board = my_pieces - enemy_pieces

    # Create image
    img_width = board_width * cell_size
    img_height = board_height * cell_size
    img = Image.new("RGB", (img_width, img_height), BLUE)
    draw = ImageDraw.Draw(img)

    for row in range(board_height):
        for col in range(board_width):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            piece = board[row, col]
            if piece == 0:
                color = WHITE
            elif piece == 1:
                color = RED
            else: # piece == -1
                color = YELLOW
            
            # Draw circle with a small padding
            padding = cell_size // 10
            draw.ellipse([x1 + padding, y1 + padding, x2 - padding, y2 - padding], fill=color, outline=color)

    return img

if __name__ == '__main__':
    sample_tensor = torch.tensor([[[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 1., 0., 1.]],

        [[0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0., 0.]]])

    # Generate image
    board_image = tensor_to_board_image(sample_tensor)
    
    # Save or show the image
    board_image.save("board_example.png")
    print("Saved an example board to board_example.png")
