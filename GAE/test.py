import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym
import copy

# experiment parameters
STEP = 100000  # total training steps
TEST_INTERVAL = 100
N_TEST_EPI = 50

#ENV = "CartPole-v0" # envs: 
#SAMPLE_NUMS = 1  # n-step boostrapping: 1, 10, 40, 80, 120, mc 
#SEED = 42  # random seed: [42, 36, 15]
# Hyper Parameters



class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out


class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def roll_out(actor_network,
             value_network,
             task,
             sample_nums,
             init_state):
    ''' Collecting data for n-step rollouts
    '''
    #task.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    v_final_state = 0  # value for final state
    state = init_state

    for _ in range(sample_nums):

        # action
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(
            ACTION_DIM, p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]

        # take action
        next_state, reward, done, _ = task.step(action)

        # collect rollouts data
        states.append(state)
        actions.append(one_hot_action)
        rewards.append(reward)

        final_state = next_state
        state = next_state

        if done:
            is_done = True
            state = task.reset()
            break

    if not is_done:
        v_final_state = value_network(
            Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states, actions, rewards, v_final_state, state


def discount_reward(r, gamma, v_final_state):
    discounted_r = np.zeros_like(r)
    running_add = v_final_state
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    
    for ENV in ["CartPole-v0"]:
        STATE_DIM = 4
        ACTION_DIM = 2
        for SEED in [42, 36, 15]:
            for SAMPLE_NUMS in [1, 10, 40, 80, 150, 200]:

                result_name = str(SEED) + "_" + str(SAMPLE_NUMS) + ".npy"

                # training env
                task = gym.make(ENV)
                task.seed(SEED)
                init_state = task.reset()

                # test env
                test_task = copy.deepcopy(task)
                test_task.seed(SEED)

                torch.manual_seed(SEED)
                np.random.seed(SEED)

                # init value network
                value_network = ValueNetwork(
                    input_size=STATE_DIM, hidden_size=40, output_size=1)
                value_network_optim = torch.optim.Adam(
                    value_network.parameters(), lr=0.001)

                # init actor network
                actor_network = ActorNetwork(STATE_DIM, 40, ACTION_DIM)
                actor_network_optim = torch.optim.Adam(
                    actor_network.parameters(), lr=0.001)

                total_test_rewards = []

                for step in range(STEP):
                    # collect data for n-step rollouts
                    states, actions, rewards, v_final_state, current_state = roll_out(
                        actor_network, value_network, task, SAMPLE_NUMS, init_state)
                    init_state = current_state
                    actions_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
                    states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))

                    # train actor network
                    actor_network_optim.zero_grad()
                    log_softmax_actions = actor_network(states_var)
                    vs = value_network(states_var).detach()
                    
                    # calculate qs
                    qs = Variable(torch.Tensor(
                        discount_reward(rewards, 0.99, v_final_state)))

                    advantages = qs - vs
                    actor_network_loss = - \
                        torch.mean(torch.sum(log_softmax_actions *
                                            actions_var, 1) * advantages)
                    actor_network_loss.backward()
                    torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
                    actor_network_optim.step()

                    # train value network
                    value_network_optim.zero_grad()
                    target_values = qs
                    values = value_network(states_var)
                    criterion = nn.MSELoss()
                    value_network_loss = criterion(values, target_values)
                    value_network_loss.backward()
                    torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
                    value_network_optim.step()

                    # Testing
                    if (step + 1) % TEST_INTERVAL == 0:
                            test_rewards = []
                            for _ in range(N_TEST_EPI):
                                
                                test_state = test_task.reset()
                                test_ep_reward = 0
                                test_done = False

                                # every test episdoe
                                while not test_done:
                                    
                                    # get action
                                    test_softmax_action = torch.exp(
                                        actor_network(Variable(torch.Tensor([test_state]))))
                                    test_action = np.argmax(
                                        test_softmax_action.data.numpy()[0])
                                    
                                    # take action
                                    test_next_state, test_reward, test_done, _ = test_task.step(
                                        test_action)
                                    
                                    test_ep_reward += test_reward
                                    test_state = test_next_state
                                    
                                    if test_done:
                                        break
                                
                                # collect rewards for this testing
                                test_rewards.append(test_ep_reward)

                            # collect rewards for every testing
                            total_test_rewards.append(test_rewards)

                            print("Step:", step+1, "Average test rewards:", np.mean(test_rewards))
                    
                np.save(result_name, np.array(total_test_rewards))


if __name__ == '__main__':
    main()
