from utils import parse_arguments, read_settings, plot_rewards, simulate
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from tqdm import tqdm
from agent import DQNAgent

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
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Agent
    agent = DQNAgent(state_size, action_size, **agent_settings)

    train_dqn(env, agent, **train_settings)


def train_dqn(env: gym.Env, agent: DQNAgent, episodes: int):
    episode_rewards = []
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

        print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(
            episode+1, episodes, total_reward, agent.epsilon))
        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            simulate(agent, env)
    plot_rewards(episode_rewards, show_result=True)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
