import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_states, n_actions, hidden=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_states, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
