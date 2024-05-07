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
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(capacity=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.policy_model = DQN(state_size, action_size)
        self.optimizer = optim.AdamW(
            self.policy_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0

    def act(self, state):
        self.steps_done += 1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.policy_model(state)
                return torch.argmax(q_values).item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32).view(-1, 1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        done_mask = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)

        Q = self.policy_model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            target_Q = reward_batch + self.gamma * \
                torch.max(self.policy_model(next_state_batch), dim=1,
                          keepdim=True)[0] * (1 - done_mask)

        loss = self.loss_fn(Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, update_frequency=10, batch_size=64, alpha=1):
        super().__init__(state_size, action_size, lr,
                         gamma, epsilon, epsilon_min, epsilon_decay, batch_size)
        self.update_frequency = update_frequency
        self.alpha = float(alpha)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32).view(-1, 1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        done_mask = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)

        Q_online = self.policy_model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_action_batch = torch.argmax(
                self.policy_model(next_state_batch), dim=1, keepdim=True)
            Q_target_next = self.target_model(
                next_state_batch).gather(1, next_action_batch)
            target_Q = reward_batch + self.gamma * \
                Q_target_next * (1 - done_mask)

        loss = self.loss_fn(Q_online, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_network()

    def update_target_network(self):
        if self.update_frequency is not None and self.steps_done % self.update_frequency == 0:
            for p1, p2 in zip(self.target_model.parameters(), self.policy_model.parameters()):
                p1.data.copy_(self.alpha * p2.data +
                              (1 - self.alpha) * p1.data)
