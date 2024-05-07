# from IPython import display
from IPython import display
import matplotlib.pyplot as plt
import torch
import yaml
import argparse
from agent import DQNAgent
from gymnasium import Env
from agent import DQNAgent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


# def simulate_notebook(agent: Agent, env: gym.Env, ax: plt.Axes) -> None:
#     state = env.reset()
#     img = ax.imshow(env.render(mode='rgb_array'))
#     done = False
#     while not done:
#         action = agent.choose_action(state)
#         img.set_data(env.render(mode='rgb_array'))
#         plt.axis('off')
#         display.display(plt.gcf())
#         display.clear_output(wait=True)
#         state, reward, done, _ = env.step(action)
#     env.close()


def simulate_notebook(agent: DQNAgent, env: Env, ax: plt.Axes) -> None:
    state, _ = env.reset()
    img = ax.imshow(env.render())
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = agent.act(state)
        img.set_data(env.render())
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    env.close()
    print(f'Total reward: {total_reward}\tSteps: {steps}')
