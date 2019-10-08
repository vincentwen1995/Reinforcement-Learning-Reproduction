import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from GAE.model import MC, A2C


def main():
    parser = argparse.ArgumentParser(description='A2C experiments')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='environment to experiment (default=CartPole-v0)')
    parser.add_argument('--same_network', type=bool, default=True, help='flag for using the same NN for both actor and critic')
    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units in the linear layers')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes (default=50)')
    parser.add_argument('--step_rollouts', type=int, default=10, help='step size for rollouts, n')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-2)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--render', default=False,
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    args = parser.parse_args()

    a2c = A2C(args)
    a2c.train()


if __name__ == '__main__':
    main()
