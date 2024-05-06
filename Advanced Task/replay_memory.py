from collections import namedtuple, deque
import random


class ReplayMemory:
    def __init__(self, capacity, transition=None):
        self.transition = namedtuple(
            'Transition', ('state', 'action', 'next_state', 'reward', 'done')) if transition is None else transition
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
