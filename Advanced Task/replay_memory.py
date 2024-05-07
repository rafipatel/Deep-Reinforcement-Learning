from collections import namedtuple, deque
import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, transition=None):
        self.transition = namedtuple(
            'Transition', ('state', 'action', 'next_state', 'reward', 'done')) if transition is None else transition
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, transition=None, eps=1e-6, alpha=0.6):
        super().__init__(capacity, transition)
        self.priorities = deque(maxlen=capacity)
        self.eps = eps  # Small value to avoid zero priority
        # Hyperparameter to control the amount of prioritization (0 - uniform, 1 - fully prioritized)
        self.alpha = alpha

    def push(self, *args):
        """Save a transition with maximum priority"""
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append(self.transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """Sample a batch of transitions, weighted by priority."""
        if len(self.memory) == 0:
            return [], []

        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(
            range(len(self.memory)), size=batch_size, p=sample_probs)
        batch = [self.memory[i] for i in indices]
        return batch, indices

    def update_priorities(self, indices, errors):
        """Update priorities based on the received TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = np.abs(error) + self.eps
