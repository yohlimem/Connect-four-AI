import torch
from torch import nn
from typing import List

class StateValue(nn.Module):
    def __init__(self, input_channels: int = 1, board_height: int = 6, board_width: int = 7, conv_layers_channels: List[int] = [32, 64], kernel_size: int = 3, fc_layer_sizes: List[int] = [256]):
        super().__init__()
        '''
        input_channels: Number of channels in the input tensor
        board_height: The height of the game board.
        board_width: The width of the game board.
        conv_layers_channels: A list of integers defining the number of output channels for each convolutional layer.
        kernel_size: The size of the convolutional kernel.
        fc_layer_sizes: A list of integers defining the size of each fully connected layer after the convolutional layers.
        '''

        # Create the convolutional layers
        conv_net_layers = []
        in_channels = input_channels
        for out_channels in conv_layers_channels:
            conv_net_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_net_layers.append(nn.ReLU())
            in_channels = out_channels
        conv_net_layers.append(nn.Flatten())
        self.conv_net = nn.Sequential(*conv_net_layers)

        # Calculate the output size of the convolutional layers to determine the input size of the fully connected layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, board_height, board_width)
            conv_out_size = self.conv_net(dummy_input).shape[1]

        # Create the normal layers
        fc_net_layers = []
        in_features = conv_out_size
        for out_features in fc_layer_sizes:
            fc_net_layers.append(nn.Linear(in_features, out_features))
            fc_net_layers.append(nn.ReLU())
            in_features = out_features
        fc_net_layers.append(nn.Linear(in_features, 1)) # Output a single value for the state value
        self.fc_net = nn.Sequential(*fc_net_layers)

    def forward(self, state: torch.Tensor):
        '''
        Forward pass through the network.
        Adds a batch dimension if the input is a single state.
        '''
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        conv_out = self.conv_net(state)
        return self.fc_net(conv_out)

    def loss(self, states: torch.Tensor, rewards: torch.Tensor):
        '''Calculates the loss of the value function'''
        values = self.forward(states).squeeze()
        return torch.mean((values - rewards)**2)
