import torch
# import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import nn  # All neural network modules
import random

# Set device3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out
     
     def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,4)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # bidrectional=True for BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        # hidden_size needs to expand both directions, *2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # concat both directions, so need to times 2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # the _ is the (hidden_state, cell_state), but not used
        out, _ = self.lstm(x, (h0, c0))
        # Apply fc to all time steps
        #out = self.fc(out)
        out = self.fc(out[:, -1, :])
        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,4)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()

class RNN_LSTM_withAttention(nn.Module):
    """
    LSTM model enhanced with multi-head attention.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads=8):
        super(RNN_LSTM_withAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = False  # Using bidirectional LSTM

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        # Adjust embed_dim for attention
        if self.bidirectional:
            self.embed_dim = hidden_size * 2
        else:
            self.embed_dim = hidden_size

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # Fully connected layer for output
        self.fc = nn.Linear(self.embed_dim, num_classes)
    def forward(self, x):
        # Move data to the appropriate device
        x = x.to(device)
        batch_size = x.size(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size * num_directions)

        # Apply multi-head attention
        # Using LSTM outputs as queries, keys, and values
        attn_output, attn_weights = self.attention(
            query=out,
            key=out,
            value=out
        )  # attn_output: (batch_size, seq_length, embed_dim)

        # Option 1: Take the output corresponding to the last time step
        attn_output = attn_output[:, -1, :]  # (batch_size, embed_dim)

        # Option 2: Aggregate over the sequence length (e.g., mean)
        # attn_output = attn_output.mean(dim=1)  # (batch_size, embed_dim)

        # Pass through the fully connected layer
        out = self.fc(attn_output)
        return out
    def sample_action(self, obs, epsilon):
        """
        Epsilon-greedy action selection.
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 4)
            return explore_action
        else:
            out = self.forward(obs)
            return out.argmax().item()
