import gym
import torch
from models import *
device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')


def main():
    print(f'{device = }')

    print(gym.envs.registry.keys())

    # base_dqn = DQN().to(device)


if __name__ == '__main__':
    main()
