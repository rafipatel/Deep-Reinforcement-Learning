from utils import parse_arguments, read_settings
import gymnasium as gym
import torch
from tqdm import tqdm
from agent import DQNAgent, DoubleDQNAgent
import collections
from logger import Logger

device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')


def main():
    print(f'{device = }')

    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    agent_settings = settings['agent']
    train_settings = settings['train']

    # Environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Agents
    dqn_agent = DQNAgent(state_size, action_size, **
                         agent_settings, device=device)

    # Logger
    logger = Logger(settings, dqn_agent.__class__.__name__, 'INM707-DRL')

    train(env, dqn_agent, logger, **train_settings)


def train(env: gym.Env, agent: DQNAgent, logger: Logger, episodes: int):
    episode_rewards = []
    recent_rewards = collections.deque(maxlen=100)
    for episode in tqdm(range(episodes), desc="Training: "):
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

        logger.log({'episode_reward': total_reward})
        if (episode + 1) % 100 == 0:
            average_reward = sum(recent_rewards) / len(recent_rewards)
            print(
                f"\rEpisode {episode + 1}/{episodes}\tAverage Score: {average_reward:.2f}")
            logger.log({'average_reward_100_epochs': average_reward})
    return episode_rewards


if __name__ == '__main__':
    main()
