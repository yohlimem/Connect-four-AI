from typing import List
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from state_value import StateValue

class Policy(nn.Module):
    def __init__(self, actions_size: int, 
                 input_channels: int = 1, board_height: int = 6, board_width: int = 7, 
                 conv_layers_channels: List[int] = [32, 64], kernel_size: int = 3, 
                 fc_layer_sizes: List[int] = [256], learning_rate: float = 0.001, ent_coef: float = 0.0):
        super().__init__()
        self.ent_coef = ent_coef
        # state value function for the critic
        self.value_function = StateValue(input_channels, board_height, board_width, conv_layers_channels, kernel_size, fc_layer_sizes)
        
        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width

        # Actor CNN Backbone
        conv_net_layers = []
        in_channels = input_channels
        for out_channels in conv_layers_channels:
            conv_net_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_net_layers.append(nn.ReLU())
            in_channels = out_channels
        
        conv_net_layers.append(nn.Flatten())
        self.conv_net = nn.Sequential(*conv_net_layers)

        # Calculate shape for fully connected layer
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, board_height, board_width)
            conv_out_size = self.conv_net(dummy).shape[1]

        
        fc_net_layers = []
        in_features = conv_out_size
        for out_features in fc_layer_sizes:
            fc_net_layers.append(nn.Linear(in_features, out_features))
            fc_net_layers.append(nn.ReLU())
            in_features = out_features
        
        fc_net_layers.append(nn.Linear(in_features, actions_size))
        self.fc_net = nn.Sequential(*fc_net_layers)

        # Optimizes both Policy and Value Function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state: torch.Tensor):
        # Reshape flat input (Batch, 42) -> (Batch, 1, 6, 7) if needed
        if state.dim() == 1:
            state = state.view(1, self.input_channels, self.board_height, self.board_width)
        elif state.dim() == 2:
            state = state.view(-1, self.input_channels, self.board_height, self.board_width)
        elif state.dim() == 3:
            state = state.unsqueeze(1)
            
        conv_out = self.conv_net(state)
        return self.fc_net(conv_out)

    def objective(self, states: torch.Tensor, actions_taken: torch.Tensor, old_log_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        
        new_log_probs = dist.log_prob(actions_taken)
        ratio = torch.exp(new_log_probs - old_log_probs)
        entropy = dist.entropy().mean()

        epsilon = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        ppo_objective = torch.min(surr1, surr2).mean()

        value_loss = self.value_function.loss(states, rewards)

        # Maximize Objective = Minimize Negative
        return -(ppo_objective + self.ent_coef * entropy - 0.5 * value_loss)

    def advantage(self, states: torch.Tensor, rewards: torch.Tensor):
        with torch.no_grad():
            # StateValue must handle reshaping internally too or be called with correct shape
            # Since StateValue is standard, we ensure shape here if needed, 
            # but usually forward() handles it.
            if states.dim() == 2:
                states_reshaped = states.view(-1, self.input_channels, self.board_height, self.board_width)
            elif states.dim() == 3:
                states_reshaped = states.unsqueeze(1)
            else:
                states_reshaped = states

            values = self.value_function(states_reshaped).detach().squeeze()
            adv = rewards - values
            return (adv - adv.mean()) / (adv.std() + 1e-8)

    def optimizer_step(self, states: torch.Tensor, actions_taken: torch.Tensor, old_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):        
        loss = self.objective(states, actions_taken, old_probs, rewards, advantages)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

    def load_from_file(self, path: str, device: str | None = None):
        if device is None:
             device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading from: {path} to {device}")
        try:
            state_dict = torch.load(path, map_location=torch.device(device))
            self.load_state_dict(state_dict, strict=True)
            print("SUCCESS: Model loaded.")
        except Exception as e:
            print(f"ERROR: {e}")
