from collections import OrderedDict
from torch import nn
from torch import optim
import torch
from state_value import StateValue
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, value_function: StateValue, state_size: int, actions_size: int, hidden_amount: int =3, layer_size: int =256, ent_coef:float =0.0):
        super().__init__()
        self.ent_coef = ent_coef

        '''
        state_size size is a number bigger than 1 that represents the amount of inputs for the network
        actions_size is a number bigger than 1 that represents the amount of outputs of the network
        hidden_amount is a number that represents the amount of hidden layers
        layer_size is a number bigger than 1 that represents the amount of neurons in a hidden layer
        '''

        # create the layers for the model
        layers_list = [("input layer", nn.Linear(state_size, layer_size)), ("input layer RELU", nn.ReLU())]
        for i in range(hidden_amount):
            layers_list.append((f"hidden layer ({i})", nn.Linear(layer_size, layer_size)))
            layers_list.append((f"hidden layer ({i}) RELU", nn.ReLU()))
        layers_list.append(("output layer", nn.Linear(layer_size, actions_size)))
        # layers_list.append(("output layer sofmax", nn.Softmax(dim=1)))

        # create the model itself
        self.model = nn.Sequential(
            OrderedDict(
                layers_list
            )
        )

        self.value_function = value_function

        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, state: torch.Tensor):
        out = self.model(state)
        return out    
    def objective(self, states: torch.Tensor, actions_taken_indices: torch.Tensor, old_log_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):
        forward = self.forward(states)
        new_probs = forward.gather(1, actions_taken_indices.long().unsqueeze(1)).squeeze() # uses the action indexes to find the correct probabilties


        dist = Categorical(logits=forward)

        new_log_probs = dist.log_prob(actions_taken_indices)
        ratio = torch.exp(new_log_probs-old_log_probs)

        entropy = dist.entropy().mean()

        biggest_grad_variance = 0.2 # epsilon

        ppo_objective = torch.mean(torch.min(ratio*advantages, torch.clip(ratio, 1-biggest_grad_variance, 1+biggest_grad_variance)*advantages))

        value_loss = self.value_function.loss(states, rewards)


        return -(ppo_objective + self.ent_coef*entropy - 0.5*value_loss) # minus for maximizing in the grad direction instead of minimizing
    
    def advantage(self, states: torch.Tensor, rewards: torch.Tensor):
        with torch.no_grad():
            target_returns = rewards
            adv = target_returns - self.value_function(states).squeeze()
            return (adv - adv.mean()) / (adv.std() + 1e-8) # Standardization
        
    def optimizer_step(self, states: torch.Tensor, actions_taken_indices: torch.Tensor, old_probs: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor):        
        objective = self.objective(states, actions_taken_indices, old_probs, rewards, advantages)
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()