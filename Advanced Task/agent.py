import torch.nn as nn
import torch
import random
import math
from replay_memory import ReplayMemory
from models import DQN
import torch.optim as optim
import numpy as np
# Define DQN agent


class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(capacity=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = self.memory.transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32).view(-1, 1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        done_mask = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)

        Q = self.model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            target_Q = reward_batch + self.gamma * \
                torch.max(self.model(next_state_batch), dim=1,
                          keepdim=True)[0] * (1 - done_mask)

        loss = self.loss_fn(Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
