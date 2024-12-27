import torch
import torch.nn.functional as F
import torch as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hybrid_FCN_Reservoir(nn.Module):
    """
    Hybrid architecture combining FCN with a reservoir layer
    The reservoir layer has fixed random weights that don't get updated during training
    """
    def __init__(self, insize, outsize, reservoir_size=1000):
        super(Hybrid_FCN_Reservoir, self).__init__()
        self.reservoir_size = reservoir_size

        # Regular trainable layers
        self.fc1 = nn.Linear(insize, 512)

        # Reservoir layer (fixed random weights)
        self.reservoir = nn.Linear(512, reservoir_size)
        # Fix the reservoir weights
        with torch.no_grad():
            # Initialize with random weights
            nn.init.xavier_uniform_(self.reservoir.weight)
            nn.init.zeros_(self.reservoir.bias)
        # Freeze the reservoir parameters
        self.reservoir.weight.requires_grad = False
        self.reservoir.bias.requires_grad = False

        # Output layers after reservoir
        self.fc2 = nn.Linear(reservoir_size, 256)
        self.fc3 = nn.Linear(256, 84)
        self.fc4 = nn.Linear(84, outsize)

    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)

        # First trainable layer
        x = F.relu(self.fc1(x))

        # Reservoir layer with fixed weights
        x = torch.tanh(self.reservoir(x))  # Using tanh as activation for reservoir

        # Output layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,4)
        else:
            out = self.forward(obs)
            return out.argmax().item()
