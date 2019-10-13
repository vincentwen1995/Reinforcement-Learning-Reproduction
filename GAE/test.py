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
from torch.distributions import Categorical

# experiment parameters
STEP = 10000  # total training steps
TEST_INTERVAL = 100
N_TEST_EPI = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 64
# ENV = "CartPole-v0" # envs:
# STEP_ROLLOUT = 1  # n-step boostrapping: 1, 10, 40, 80, 120, mc
# SEED = 42  # random seed: [42, 36, 15]
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
        out = F.log_softmax(self.fc3(out), dim=-1)
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
             env,
             step_rollout,
             init_state,
             action_dim):
    ''' Collecting data for n-step rollouts
    '''
    # env.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    v_final_state = 0  # value for final state
    state = init_state

    for _ in range(step_rollout):

        # action
        log_softmax_action = actor_network(torch.tensor([state], dtype=torch.float, device=DEVICE))
        softmax_action = torch.exp(log_softmax_action)

        action_dist = Categorical(softmax_action)
        action = action_dist.sample()
        # action = np.random.choice(
        # action_dim, p=softmax_action.cpu().data.numpy()[0])
        # one_hot_action = [int(k == action) for k in range(action_dim)]
        one_hot_action = [int(k == action.item()) for k in range(action_dim)]

        # take action
        next_state, reward, done, _ = env.step(action.item())

        # collect rollouts data
        states.append(state)
        actions.append(one_hot_action)
        rewards.append(reward)

        final_state = next_state
        state = next_state

        if done:
            is_done = True
            state = env.reset()
            break

    if not is_done:
        v_final_state = value_network(
            torch.tensor([final_state], dtype=torch.float, device=DEVICE)).cpu().data.numpy()

    return states, actions, rewards, v_final_state, state


def discount_reward(r, gamma, v_final_state):
    discounted_r = np.zeros_like(r)
    running_add = v_final_state
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():

    # for ENV, STATE_DIM, ACTION_DIM, STEP_ROLLOUTS in zip(["CartPole-v0", "MountainCar-v0", 'Acrobot-v1'], [4, 2, 6], [2, 3, 3], [[1, 10, 40, 80, 150, 200], [1, 10, 40, 80, 150, 200], [1, 30, 60, 100, 300, 500]]):
    # for ENV, STATE_DIM, ACTION_DIM, STEP_ROLLOUTS in zip(["MountainCar-v0"], [2], [3], [[1, 10, 40, 80, 150, 200]]):
    for ENV, STATE_DIM, ACTION_DIM, STEP_ROLLOUTS in zip(['Acrobot-v1'], [6], [3], [[1, 30, 60, 100, 300, 500]]):
        # STATE_DIM = 4
        # ACTION_DIM = 2
        print('Env: {}'.format(ENV))
        for SEED in [42, 36, 15]:
            print('Seed: {}'.format(SEED))
            for STEP_ROLLOUT in STEP_ROLLOUTS:
                print('Step Rollout: {}'.format(STEP_ROLLOUT))

                result_file_name = 'results/' + ENV + '_' + str(SEED) + "_" + str(STEP_ROLLOUT) + ".npy"

                # training env
                env = gym.make(ENV)
                env.seed(SEED)
                init_state = env.reset()

                # test env
                test_env = copy.deepcopy(env)
                test_env.seed(SEED)

                torch.manual_seed(SEED)
                np.random.seed(SEED)

                # init value network
                value_network = ValueNetwork(
                    input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=1)
                value_network.to(DEVICE)
                value_network_optim = torch.optim.Adam(
                    value_network.parameters(), lr=0.001)

                # init actor network
                actor_network = ActorNetwork(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, action_size=ACTION_DIM)
                actor_network.to(DEVICE)
                actor_network_optim = torch.optim.Adam(
                    actor_network.parameters(), lr=0.001)

                total_test_rewards = []

                for step in range(STEP):
                    # collect data for n-step rollouts
                    states, actions, rewards, v_final_state, current_state = roll_out(
                        actor_network, value_network, env, STEP_ROLLOUT, init_state, ACTION_DIM)
                    init_state = current_state
                    actions_var = torch.tensor(actions, dtype=torch.float, device=DEVICE).view(-1, ACTION_DIM)
                    states_var = torch.tensor(states, dtype=torch.float, device=DEVICE).view(-1, STATE_DIM)

                    # train actor network
                    actor_network_optim.zero_grad()
                    log_softmax_actions = actor_network(states_var)
                    vs = value_network(states_var).detach()

                    # calculate qs
                    qs = torch.tensor(
                        discount_reward(rewards, 0.99, v_final_state), dtype=torch.float, device=DEVICE)

                    advantages = qs - vs
                    actor_network_loss = - \
                        torch.mean(torch.sum(log_softmax_actions *
                                             actions_var, dim=1) * advantages)
                    actor_network_loss.backward()
                    torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
                    actor_network_optim.step()

                    # train value network
                    value_network_optim.zero_grad()
                    target_values = qs
                    values = value_network(states_var)
                    criterion = nn.MSELoss()
                    value_network_loss = criterion(values, target_values.unsqueeze(1))
                    value_network_loss.backward()
                    torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
                    value_network_optim.step()

                    # Testing
                    with torch.no_grad():
                        if (step + 1) % TEST_INTERVAL == 0:
                            test_rewards = []
                            for _ in range(N_TEST_EPI):

                                test_state = test_env.reset()
                                test_ep_reward = 0
                                test_done = False

                                # every test episdoe
                                while not test_done:

                                    # get action
                                    test_softmax_action = torch.exp(
                                        actor_network(torch.tensor([test_state], dtype=torch.float, device=DEVICE)))
                                    test_action = np.argmax(
                                        test_softmax_action.cpu().numpy()[0])

                                    # take action
                                    test_next_state, test_reward, test_done, _ = test_env.step(
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

                np.save(result_file_name, np.array(total_test_rewards))


if __name__ == '__main__':
    main()
