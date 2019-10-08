import argparse
import gym
from gym import spaces
import numpy as np
from itertools import count
from collections import namedtuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils.utils import BaseMethod, get_space_shape

EPSILON = np.finfo(np.float32).eps.item()


class Policy(nn.Module):

    def __init__(self, num_obs, num_actions, same_network, num_hidden):
        super(Policy, self).__init__()

        self.actor_linear = nn.Linear(num_obs, num_hidden)
        self.actor_out = nn.Linear(num_hidden, num_actions)

        if same_network:
            self.critic_linear = self.actor_linear
        else:
            self.critic_linear = nn.Linear(num_obs, num_hidden)
        self.critic_out = nn.Linear(num_hidden, 1)

    def forward(self, x):

        actor_logits = F.relu(self.actor_linear(x))
        critic_logits = F.relu(self.critic_linear(x))

        action_prob = F.softmax(self.actor_out(actor_logits), dim=-1)
        state_values = self.critic_out(critic_logits)

        return action_prob, state_values


class MC(BaseMethod):

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.args.device)
        action_prob, state_value = self.model(state)

        action_dist = Categorical(action_prob)
        #print(action_prob)
        action = action_dist.sample()

        #print(action.item())
        #print(action_dist.log_prob(action))
        #print(state_value)
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

        returns = torch.tensor(returns)
        #print("returns: ", returns)
        #returns = (returns - returns.mean()) / (returns.std() + EPSILON)
        
        final_v = self.trajectory[-1][1][-1]
        #print(final_v)
        for step, ret, i in zip(self.trajectory, returns, range(len(returns))):
            _, log_prob, value = step[1]
            advantage = ret + self.args.gamma**(self.args.step_rollouts-i) * final_v.item() - value.item()
            #print("ret: ", ret)
            #advantage = ret - value.item()


            # calculate actor (policy) loss
            actor_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([ret], device=self.args.device)))

        # sum up all the values of actor_losses and critic_losses
        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()

        # perform backprop
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(),0.5)

    def train(self):

        env = gym.make(self.args.env)
        env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.model = Policy(
            num_obs=get_space_shape(env.observation_space),
            num_actions=get_space_shape(env.action_space),
            same_network=self.args.same_network,
            num_hidden=self.args.num_hidden
        )
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        running_reward = 10

        for i_episode in np.arange(self.args.episodes):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0
            done = False

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            
            while not done:
                
                self.trajectory = []
                for t in np.arange(self.args.step_rollouts):

                    # select action from policy
                    action, log_prob, state_value = self.select_action(state)

                    # take the action
                    next_state, reward, done, _ = env.step(action)
                    self.trajectory.append((state, (action, log_prob, state_value), reward))

                    ep_reward += reward

                    if done:
                        break

                    state = next_state

                    if self.args.render:
                        env.render()

                    


                # update cumulative reward
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

                # perform backprop
                optimizer.zero_grad()
                self.update()
                optimizer.step()
            
            # log results
            if i_episode % self.args.log_interval == 0:

                test_env = copy.copy(env)
                test_rewards = []

                for _ in range(10):
                    test_state = env.reset()
                    test_ep_reward = 0
                    test_done = False

                    while not test_done:
                        test_action, _, _ = self.select_action(test_state)
                        test_state, test_reward, test_done, _ = env.step(test_action)
                        test_ep_reward += test_reward
                        if test_done:
                            break
                    test_rewards.append(test_ep_reward)
                test_avg_reward = np.mean(test_rewards)
                print('Episode {}\tLast reward: {:.3f}\tAverage reward: {:.3f}'.format(
                    i_episode, ep_reward, test_avg_reward))

            # check if we have "solved" the cart pole problem
            if test_avg_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(test_avg_reward, t))
                break


class A2C(BaseMethod):

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.args.device)
        action_prob, state_value = self.model(state)

        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        return action.item(), action_dist.log_prob(action), state_value

    def update(self):
        actor_losses = []
        critic_losses = []
        returns = []

        # calculate cumulated discounted rewards
        ret = 0
        for step in self.trajectory[::-1]:
            reward = step[2]
            ret = reward + self.args.gamma * ret
            returns.insert(0, ret)

        returns = torch.tensor(returns)
        

        final_v = self.trajectory[-1][1][-1]
        for step, ret, i in zip(self.trajectory, returns, range(len(returns))):
            _, log_prob, value = step[1]
            q = ret + self.args.gamma**(self.args.step_rollouts-i) * final_v.item()
            v = value.item()
            advantage = q - v 
            actor_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([ret], device=self.args.device)))

        # sum up all the values of actor_losses and critic_losses
        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()

        # perform backprop
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(),0.5)

    def train(self):

        env = gym.make(self.args.env)
        env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.model = Policy(
            num_obs=get_space_shape(env.observation_space),
            num_actions=get_space_shape(env.action_space),
            same_network=self.args.same_network,
            num_hidden=self.args.num_hidden
        )
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        for i_episode in np.arange(self.args.episodes):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0
            done = False

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            
            while not done:
                
                self.trajectory = []
                # start n-step rollouts 
                for _ in np.arange(self.args.step_rollouts):

                    # select action from policy
                    action, log_prob, state_value = self.select_action(state)

                    # take the action
                    next_state, reward, done, _ = env.step(action)
                    self.trajectory.append((state, (action, log_prob, state_value), reward))

                    ep_reward += reward

                    if done:
                        break

                    state = next_state

                # perform backprop
                optimizer.zero_grad()
                self.update()
                optimizer.step()
            
            # log results
            if i_episode % self.args.log_interval == 0:
                
                # tests
                test_env = copy.copy(env)
                test_rewards = []

                for _ in range(10):
                    test_state = env.reset()
                    test_ep_reward = 0
                    test_done = False

                    while not test_done:
                        test_action, _, _ = self.select_action(test_state)
                        test_state, test_reward, test_done, _ = env.step(test_action)
                        test_ep_reward += test_reward
                        
                        if test_done:
                            break

                    test_rewards.append(test_ep_reward)
                
                test_avg_reward = np.mean(test_rewards)
                
                print('Episode {}\tLast training reward: {:.3f}\tAverage test reward: {:.3f}'.format(
                    i_episode, ep_reward, test_avg_reward))

            # check if we have "solved" the cart pole problem
            if test_avg_reward > env.spec.reward_threshold:
                print("Solved! Average test reward is now {}!".format(test_avg_reward))
                break
            
