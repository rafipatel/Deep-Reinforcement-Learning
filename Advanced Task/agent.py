import torch.nn as nn
import torch
import random
import math
from replay_memory import ReplayMemory, PrioritizedReplayMemory
from models import DQN
import torch.optim as optim
import numpy as np

# Define a DQN agent class to manage the Deep Q-Network (DQN) behavior.


class DQNAgent:
    # Constructor to initialize the DQN agent with various parameters for configuration.
    def __init__(self, state_size: int, action_size: int, hidden_units=128, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, use_prioritized_replay=False, replay_memory_alpha=0.6):
        # Store input parameters and set up the environment and model.
        self.state_size = state_size  # The size of an input state to the DQN.
        # The number of possible actions the agent can take.
        self.action_size = action_size
        self.hidden_units = hidden_units  # Number of hidden units in the DQN.
        # Whether to use prioritized replay memory.
        self.use_prioritized_replay = use_prioritized_replay
        self.replay_memory_alpha = replay_memory_alpha
        # Initialize the appropriate replay memory type based on the above flag.
        self.memory = PrioritizedReplayMemory(
            10000, alpha=replay_memory_alpha) if use_prioritized_replay else ReplayMemory(10000)
        self.gamma = gamma  # Discount factor for future rewards.
        # Initial probability for taking a random action (exploration).
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # Minimum exploration probability.
        # Decay rate for exploration probability.
        self.epsilon_decay = epsilon_decay
        # Number of experiences to sample from memory during training.
        self.batch_size = batch_size
        # Initialize the policy model (DQN) with specified state and action sizes.
        self.policy_model = DQN(state_size, action_size,
                                hidden_units)
        # Set up the optimizer with AdamW algorithm (variation of Adam better suited for weight decay).
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        # Mean Squared Error Loss to measure prediction accuracy.
        self.loss_fn = nn.MSELoss()
        # Counter for steps completed (for logging or modifying behavior over time).
        self.steps_done = 0
        self.lr = lr

    # Function to select an action based on current state.
    def act(self, state):
        self.steps_done += 1  # Increment step counter.
        # Decide between exploration and exploitation.
        if np.random.rand() <= self.epsilon:
            # Return a random action (explore).
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():  # Temporarily set all the requires_grad flag to false.
                # Convert state to tensor.
                state = torch.tensor(
                    state, dtype=torch.float32)
                # Get Q-values for all actions from the policy model.
                q_values = self.policy_model(state)
                # Choose the action with the highest Q-value (exploit).
                return torch.argmax(q_values).item()

    # Method to store an experience (transition) in replay memory.
    def remember(self, state, action, next_state, reward, done):
        # Push the experience to the memory.
        self.memory.push(state, action, next_state, reward, done)

    # Method to train the model with experiences sampled from memory.
    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Do not train if there aren't enough memories for a batch.
        # Sample a batch of transitions.
        transitions, indices = self.memory.sample(self.batch_size)
        # Unzip transitions into separate variables.
        batch = self.memory.transition(*zip(*transitions))

        # Convert batches of experiences to tensors.
        state_batch = torch.tensor(
            batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32).view(-1, 1)
        next_state_batch = torch.tensor(
            batch.next_state, dtype=torch.float32)
        done_mask = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)

        # Compute the Q values for current states (Q-learning).
        Q = self.policy_model(state_batch).gather(1, action_batch)
        with torch.no_grad():  # Disable gradient calculation.
            # Calculate the target Q value for next states.
            target_Q = reward_batch + self.gamma * \
                torch.max(self.policy_model(next_state_batch),
                          dim=1, keepdim=True)[0] * (1 - done_mask)

        # Calculate the loss between current Q values and target Q values.
        loss = self.loss_fn(Q, target_Q)

        self.update_buffer_priorities(indices, target_Q, Q)

        # Perform a backpropagation pass to update weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate (epsilon), ensuring it does not fall below the minimum.
        self.decay_epsilon()

    def update_buffer_priorities(self, indices, target_Q, Q):
        if self.use_prioritized_replay:  # Check if using prioritized replay.
            # Update priorities based on TD errors, to prioritize important experiences.
            td_errors = (target_Q - Q).squeeze().detach().abs().numpy()
            self.memory.update_priorities(indices, td_errors)

    # Method to save the current state of the model to a file.
    def save_state(self, path):
        state_dict = {
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'hidden_units': self.hidden_units,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'use_prioritized_replay': self.use_prioritized_replay,
            'replay_memory_alpha': self.replay_memory_alpha
        }

        torch.save(state_dict, path)

    # Method to load the state of the model from a file.
    def load_state(self, path):
        checkpoint = torch.load(path)  # Load data from the file.
        # Restore model parameters.
        self.policy_model.load_state_dict(
            checkpoint['policy_model_state_dict'])
        # Restore optimizer parameters.
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def decay_epsilon(self):
        # Update exploration probability.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DoubleDQNAgent class, extending DQNAgent, to use Double DQN strategy (helps reduce overestimations).


class DoubleDQNAgent(DQNAgent):
    # Constructor with additional parameters specific to Double DQN.
    def __init__(self, state_size: int, action_size: int, hidden_units=128, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 update_frequency=10, batch_size=64, alpha=1, use_prioritized_replay=False, replay_memory_alpha=0.6):
        super().__init__(state_size, action_size, hidden_units, lr, gamma, epsilon, epsilon_min,
                         epsilon_decay, batch_size, use_prioritized_replay, replay_memory_alpha)
        # Frequency at which to update the target model.
        self.update_frequency = update_frequency
        # Factor for interpolating between the target and policy models.
        self.alpha = float(alpha)
        # Secondary (target) model for Double DQN.
        self.target_model = DQN(state_size, action_size,
                                hidden_units)
        # Initialize target model weights to policy model's weights.
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()  # Set the target model to evaluation mode.

    # Overridden replay method to incorporate Double DQN logic.
    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Avoid training if there's not enough data.
        # Sample experiences.
        transitions, indices = self.memory.sample(self.batch_size)
        # Unzip transitions.
        batch = self.memory.transition(*zip(*transitions))

        # Convert experience batches to tensors.
        state_batch = torch.tensor(
            batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32).view(-1, 1)
        next_state_batch = torch.tensor(
            batch.next_state, dtype=torch.float32)
        done_mask = torch.tensor(batch.done, dtype=torch.float32).view(-1, 1)

        # Compute Q values using the policy model.
        Q_online = self.policy_model(state_batch).gather(1, action_batch)
        with torch.no_grad():  # Disable gradients for the next part.
            # Double DQN adjustment: use the policy model to select actions and the target model to evaluate them.
            next_action_batch = torch.argmax(
                self.policy_model(next_state_batch), dim=1, keepdim=True)
            Q_target_next = self.target_model(
                next_state_batch).gather(1, next_action_batch)
            target_Q = reward_batch + self.gamma * \
                Q_target_next * (1 - done_mask)

        # Calculate loss and perform backpropagation.
        loss = self.loss_fn(Q_online, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_buffer_priorities(indices, target_Q, Q_online)

        self.decay_epsilon()  # Decay exploration rate.

        # Periodically update the weights of the target network to slowly track the policy network.
        if self.update_frequency is not None and self.steps_done % self.update_frequency == 0:
            self.update_target_network(self.alpha)

    # Method to update the target network by interpolating between its weights and the policy model's weights.
    def update_target_network(self, alpha):
        for p1, p2 in zip(self.target_model.parameters(), self.policy_model.parameters()):
            # Update target model parameters.
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    def save_state(self, path):
        state_dict = {
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'hidden_units': self.hidden_units,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'use_prioritized_replay': self.use_prioritized_replay,
            'replay_memory_alpha': self.replay_memory_alpha,
            'target_model_state_dict': self.target_model.state_dict(),
            'update_frequency': self.update_frequency,
            'alpha': self.alpha
        }

        torch.save(state_dict, path)

    # Overridden method to load state and then update target network weights to ensure consistency.
    def load_state(self, path):
        super().load_state(path)  # Call base class load.
        # Ensure target network is synchronized with policy network.
        self.update_target_network(alpha=1)
