import os
import argparse
import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from utils.utils import BaseMethod, get_space_shape
# from torchviz import make_dot, make_dot_from_trace

EPSILON = np.finfo(np.float32).eps.item()

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Policy(nn.Module):
    """Actor-critic network.
    """

    def __init__(self, num_obs, num_actions, same_network, num_hidden, action_cont):
        super(Policy, self).__init__()
        self.action_cont = action_cont
        self.same_network = same_network

        self.actor_linear = nn.Linear(num_obs, num_hidden)
        if self.action_cont:
            self.actor_out = nn.Linear(num_hidden, num_actions * 2)
        else:
            self.actor_out = nn.Linear(num_hidden, num_actions)

        if not self.same_network:
            self.critic_linear = nn.Linear(num_obs, num_hidden)

        self.critic_out = nn.Linear(num_hidden, 1)

    def forward(self, x):

        actor_logits = F.relu(self.actor_linear(x))
        if self.same_network:
            critic_logits = F.relu(self.actor_linear(x))
        else:
            critic_logits = F.relu(self.critic_linear(x))

        if self.action_cont:
            action_output = F.relu(self.actor_out(actor_logits))
        else:
            action_output = F.softmax(self.actor_out(actor_logits), dim=-1)
        state_values = self.critic_out(critic_logits)

        return action_output, state_values


class MC(BaseMethod):
    """Monte-carlo method for GAE.
    """

    def __init__(self, args):
        super(MC, self).__init__()
        self.args = args

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.args.device)
        action_output, state_value = self.model(state)

        if self.action_cont:
            action = []
            log_probs = []
            means, stds = action_output[:self.nA], action_output[self.nA:]
            # stds = stds.abs()
            for i_dists in np.arange(self.nA):
                tmp_dist = Normal(means[i_dists], stds[i_dists] + EPSILON)
                action_ = tmp_dist.sample()
                action.append(action_)
                log_probs.append(tmp_dist.log_prob(action_).to(self.args.device))
            return torch.stack(action).cpu(), torch.stack(log_probs), state_value

        else:
            action_dist = Categorical(action_output)
            action = action_dist.sample()

            return action.item(), action_dist.log_prob(action), state_value

    def update(self):
        actor_losses = []
        critic_losses = []
        returns = []

        ret = 0
        for step in self.trajectory[::-1]:
            reward = step[2]
            ret = reward + self.args.gamma * ret
            returns.insert(0, ret)

        returns = torch.tensor(returns, device=self.args.device)
        returns = (returns - returns.mean()) / (returns.std() + EPSILON)
        for step, ret in zip(self.trajectory, returns):
            _, log_prob, value = step[1]
            advantage = ret - value
            # calculate actor (policy) loss
            actor_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([ret], device=self.args.device)))

        # sum up all the values of actor_losses and critic_losses
        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()

        # dot = make_dot(loss)
        # dot.format = 'pdf'
        # dot.render('graph')
        # perform backprop
        loss.backward()

    def train(self):

        env = gym.make(self.args.env)
        env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.nS = get_space_shape(env.observation_space)
        self.nA = get_space_shape(env.action_space)

        if type(env.action_space) is gym.spaces.Box:
            self.action_cont = True
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
        else:
            self.action_cont = False

        self.model = Policy(
            num_obs=self.nS,
            num_actions=self.nA,
            same_network=self.args.same_network,
            num_hidden=self.args.num_hidden,
            action_cont=self.action_cont
        )

        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        if env.spec.reward_threshold and env.spec.reward_threshold < 0.:
            running_reward = -200.0
        elif env.spec.reward_threshold and env.spec.reward_threshold >= 0.:
            running_reward = 10.0
        else:
            running_reward = 0.

        for i_episode in np.arange(self.args.episodes):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            self.trajectory = []
            for t in np.arange(env.spec.max_episode_steps):

                # select action from policy
                action, log_prob, state_value = self.select_action(state)
                # take the action
                next_state, reward, done, _ = env.step(action)
                self.trajectory.append((state, (action, log_prob, state_value), reward))
                state = next_state

                if self.args.render:
                    env.render()

                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            optimizer.zero_grad()
            self.update()
            optimizer.step()

            # log results
            if i_episode % self.args.log_interval == 0:
                print('Episode {}\tLast reward: {:.3f}\tAverage reward: {:.3f}'.format(
                    i_episode, ep_reward, running_reward))

            # check if we have "solved" the cart pole problem
            if env.spec.reward_threshold and running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {:3f} time steps!".format(running_reward, t + 1))
                break
