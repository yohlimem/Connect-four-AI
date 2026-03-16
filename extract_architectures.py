
import torch
import os
import json
import glob
from collections import OrderedDict

def get_model_architecture(model_path):
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        conv_layers_channels = []
        fc_layer_sizes = []
        
        is_cnn = any("conv" in key for key in state_dict.keys())
        
        if is_cnn:
            # Extract conv layers
            for key, value in state_dict.items():
                if "conv_net" in key and "weight" in key and "value" not in key:
                    conv_layers_channels.append(value.shape[0])
            
            # Extract fc layers
            for key, value in state_dict.items():
                if "fc_net" in key and "weight" in key and "value" not in key:
                    fc_layer_sizes.append(value.shape[0])
            # last layer is the output layer, so we dont need it.
            if fc_layer_sizes:
                fc_layer_sizes.pop()

            return {"type": "cnn", "conv_layers_channels": conv_layers_channels, "fc_layer_sizes": fc_layer_sizes}
        else:
            # Extract rnn layers
            hidden_amount = 0
            layer_size = 0
            for key, value in state_dict.items():
                if "hidden" in key and "weight" in key:
                    hidden_amount += 1
                    layer_size = value.shape[0]

            return {"type": "rnn", "hidden_amount": hidden_amount, "layer_size": layer_size}
            
    except Exception as e:
        print(f"Error inspecting {model_path}: {e}")
        return None
def create_architectures_json():
    architectures = {}
    bot_files = glob.glob("Saves/bots/beat_alpha_beta_bot.pth") + glob.glob("Saves/bots/connect4_parallel_iter_2*.pth") + glob.glob("Saves/unused/*.pth")
    
    for model_path in bot_files:
        print(f"Extracting architecture from {model_path}")
        arch = get_model_architecture(model_path)
        if arch:
            architectures[model_path] = arch
            
    with open("architectures.json", "w") as f:
        json.dump(architectures, f, indent=4)
        
    print("Architectures extracted and saved to architectures.json")

if __name__ == "__main__":
    create_architectures_json()