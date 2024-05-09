import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from tqdm import tqdm
from agent import DQNAgent, DoubleDQNAgent
import os
import collections
import numpy as np
from sklearn.model_selection import ParameterGrid
from logger import Logger


def train_dqn(env: gym.Env, agent: DQNAgent, episodes: int, logger: Logger):
    episode_rewards = []
    recent_rewards = collections.deque(maxlen=100)
    for episode in tqdm(range(episodes), desc="Training: ", leave=False):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward
            agent.replay()

        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        logger.log({'total_reward': total_reward})
        if (episode + 1) % 100 == 0:
            average_rewards = sum(recent_rewards) / len(recent_rewards)
            logger.log({'average_reward_100_epochs': average_rewards})
            print(
                f"\rEpisode {episode + 1}/{episodes}\tAverage Score: {average_rewards}")
    return episode_rewards


train_settings = {'episodes': 2000}
print(f'{train_settings = }')
# Environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
checkpoint_folder = 'checkpoints'
os.makedirs(checkpoint_folder, exist_ok=True)
# Agent
dqn_agent_settings = {'lr': [1e-3, 1e-4],
                      'gamma': [0.95, 0.99, 0.999],
                      'epsilon': [1.0],
                      'epsilon_min': [0.01],
                      'epsilon_decay': [0.99, 0.95, 0.9],
                      'batch_size': [64],
                      'hidden_units': [64]
                      }
dqn_params = ParameterGrid(dqn_agent_settings)
dqn_checkpoint_path = os.path.join(checkpoint_folder, 'DQNAgent.pth')

max_rewards = -np.inf
for param in tqdm(dqn_params):
    print(param)
    dqn_agent = DQNAgent(state_size, action_size, **param)
    logger = Logger(param, dqn_agent.__class__.__name__, 'INM707-DRL')

    dqn_rewards = train_dqn(env, dqn_agent, **train_settings, logger=logger)

    if x := np.mean(dqn_rewards) > max_rewards:
        max_rewards = x
        best_dqn_config = param
        dqn_agent.save_state(dqn_checkpoint_path)
dqn_priority_checkpoint_path = os.path.join(
    checkpoint_folder, 'DQN_Priority_Agent.pth')

max_rewards = -np.inf
for param in tqdm(dqn_params):
    dqn_priority_agent = DQNAgent(state_size, action_size, **param,
                                  use_prioritized_replay=True)
    logger = Logger(param, dqn_priority_agent.__class__.__name__ +
                    '_Priority', 'INM707-DRL')

    print(param)

    dqn_priority_rewards = train_dqn(
        env, dqn_priority_agent, **train_settings, logger=logger)

    if x := np.mean(dqn_priority_rewards) > max_rewards:
        max_rewards = x
        best_dqn_priority_config = param
        dqn_priority_agent.save_state(dqn_priority_checkpoint_path)
ddqn_agent_settings = {**dqn_agent_settings,
                       'alpha': [5e-1, 1e-1, 1e-3],
                       'hidden_units': 64,
                       'update_frequency': [4, 10]}
ddqn_params = ParameterGrid(ddqn_agent_settings)
ddqn_checkpoint_path = os.path.join(checkpoint_folder, 'DoubleDQNAgent.pth')

max_rewards = -np.inf
for param in tqdm(ddqn_params):
    print(param)
    ddqn_agent = DoubleDQNAgent(state_size, action_size,
                                **param)
    logger = Logger(param, ddqn_agent.__class__.__name__, 'INM707-DRL')

    ddqn_rewards = train_dqn(env, ddqn_agent, **train_settings, logger=logger)

    if x := np.mean(ddqn_rewards) > max_rewards:
        max_rewards = x
        best_ddqn_config = param
        ddqn_agent.save_state(ddqn_checkpoint_path)
ddqn_priority_checkpoint_path = os.path.join(
    checkpoint_folder, 'DoubleDQN_Priority_Agent.pth')

max_rewards = -np.inf
for param in tqdm(ddqn_params):
    print(param)
    ddqn_priority_agent = DoubleDQNAgent(state_size, action_size,
                                         **param, use_prioritized_replay=True)
    logger = Logger(
        param, ddqn_priority_agent.__class__.__name__, 'INM707-DRL')

    ddqn_priority_rewards = train_dqn(
        env, ddqn_priority_agent, **train_settings, logger=logger)

    if x := np.mean(ddqn_priority_rewards) > max_rewards:
        max_rewards = x
        best_ddqn_priority_config = param
        ddqn_priority_agent.save_state(ddqn_priority_checkpoint_path)

print(f'{best_dqn_config = }\n'
      f'{best_ddqn_config = }\n'
      f'{best_dqn_priority_config = }\n'
      f'{best_ddqn_priority_config = }\n')
# def moving_average(a, n=100):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# _, ax = plt.subplots(1, 1, figsize=(20, 12))
# ax.plot(dqn_rewards, label='DQN')
# ax.plot(dqn_priority_rewards, label='DQN Priority')
# ax.plot(ddqn_rewards, label='DoubleDQN')
# ax.plot(ddqn_priority_rewards, label='DoubleDQN Priority')
# ax.legend()
# ax.set_ylabel("Total Reward")
# ax.set_xlabel("Episode")
# plt.show()
# _, ax = plt.subplots(1, 1, figsize=(20, 12))
# ax.plot(moving_average(dqn_rewards), label='DQN')
# ax.plot(moving_average(ddqn_rewards), label='DoubleDQN')
# ax.plot(moving_average(dqn_priority_rewards), label='DQN Priority')
# ax.plot(moving_average(ddqn_priority_rewards), label='DoubleDQN Priority')
# ax.legend()
# ax.set_ylabel("Total Reward")
# ax.set_xlabel("Episode")
# plt.show()
# _, ax = plt.subplots(1, 1, figsize=(10, 8))
# simulate_notebook(dqn_agent, env, ax)
# _, ax = plt.subplots(1, 1, figsize=(10, 8))
# simulate_notebook(dqn_priority_agent, env, ax)
# _, ax = plt.subplots(1, 1, figsize=(10, 8))
# simulate_notebook(ddqn_priority_agent, env, ax)
# _, ax = plt.subplots(1, 1, figsize=(10, 8))
# simulate_notebook(ddqn_agent, env, ax)
