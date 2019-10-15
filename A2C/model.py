import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from torch.distributions import Categorical, Normal

STEP = 10000  # total training steps
TEST_INTERVAL = 100
N_TEST_EPI = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 64
GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()


class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size, is_cont=False, action_bound=1.):
        super(ActorNetwork, self).__init__()
        self.is_cont = is_cont
        self.low_action_bound = -action_bound
        self.high_action_bound = action_bound
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if self.is_cont:
            self.fc3 = nn.Linear(hidden_size, action_size * 2)
        else:
            self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        if self.is_cont:
            out = self.fc3(out)
        else:
            out = F.log_softmax(self.fc3(out), dim=-1)
        return out


class CriticNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()
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
             action_dim,
             is_cont=False):
    ''' Collecting data for n-step rollouts
    '''
    # env.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    v_final_state = 0  # value for final state
    state = init_state

    for _ in np.arange(step_rollout):

        # action
        if is_cont:
            action_output = actor_network(torch.tensor([state], dtype=torch.float, device=DEVICE))
            mean, std = action_output[:, :1], action_output[:, 1:]
            std = std.abs()
            action_dist = Normal(mean, std + EPS)
            action = [action_dist.sample().item()]
            one_hot_action = action
        else:
            log_softmax_action = actor_network(torch.tensor([state], dtype=torch.float, device=DEVICE))
            softmax_action = torch.exp(log_softmax_action)

            action_dist = Categorical(softmax_action)
            action = action_dist.sample().item()
            one_hot_action = [int(k == action) for k in np.arange(action_dim)]

        # take action
        next_state, reward, done, _ = env.step(action)

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
    for t in reversed(np.arange(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def train():
    # for env_name, state_dim, action_dim, step_rollouts in \
            # zip(['CartPole-v0', 'Pendulum-v0'], [4, 3], [2, 1], [[1, 10, 40, 80, 150, 200], [1, 10, 40, 80, 150, 200]]):
    for env_name, state_dim, action_dim, step_rollouts in \
            zip(['Pendulum-v0'], [3], [1], [[1, 10, 40, 80, 150, 200]]):
        # for env_name, state_dim, action_dim, step_rollouts in \
            # zip(['CartPole-v0'], [4], [2], [[1, 10, 40, 80, 150, 200]]):

        print('Env: {}'.format(env_name))
        if env_name == 'CartPole-v0':
            is_cont = False
        else:
            is_cont = True
        for seed in [42, 36, 15]:
            print('Seed: {}'.format(seed))
            for step_rollout in step_rollouts:
                print('Step Rollout: {}'.format(step_rollout))
                result_file_name = 'results/' + env_name + '_' + str(seed) + '_' + str(step_rollout) + '.npy'
                # train env
                env = gym.make(env_name)
                env.seed(seed)
                init_state = env.reset()
                # test env
                test_env = copy.deepcopy(env)
                test_env.seed(seed)

                torch.manual_seed(seed)
                np.random.seed(seed)

                # init value network
                value_network = CriticNetwork(
                    input_size=state_dim, hidden_size=HIDDEN_SIZE, output_size=1)
                value_network.to(DEVICE)
                value_network_optim = torch.optim.Adam(
                    value_network.parameters(), lr=0.001)

                # init actor network
                actor_network = ActorNetwork(input_size=state_dim, hidden_size=HIDDEN_SIZE, action_size=action_dim, is_cont=is_cont)
                actor_network.to(DEVICE)
                actor_network_optim = torch.optim.Adam(
                    actor_network.parameters(), lr=0.001)

                total_test_rewards = []

                for step in np.arange(STEP):
                    # collect data for n-step rollouts
                    states, actions, rewards, v_final_state, current_state = roll_out(
                        actor_network, value_network, env, step_rollout, init_state, action_dim, is_cont=is_cont)
                    init_state = current_state
                    actions_var = torch.tensor(actions, dtype=torch.float, device=DEVICE).view(-1, action_dim)
                    states_var = torch.tensor(states, dtype=torch.float, device=DEVICE).view(-1, state_dim)

                    # train actor network
                    actor_network_optim.zero_grad()
                    if is_cont:
                        action_outputs = actor_network(states_var)
                        means, stds = action_outputs[:, :1], action_outputs[:, 1:]
                        vs = value_network(states_var).detach()
                        qs = torch.tensor(
                            discount_reward(rewards, GAMMA, v_final_state), dtype=torch.float, device=DEVICE)
                        advantages = qs - vs
                        pdf = 1. / torch.sqrt(2. * np.pi * stds.pow(2)) * torch.exp(- (actions_var - means).pow(2) / (2. * stds.pow(2)))
                        log_pdf = torch.log(pdf + EPS)
                        actor_network_loss = - (log_pdf * advantages).mean()

                    else:
                        log_softmax_actions = actor_network(states_var)
                        vs = value_network(states_var).detach()

                        # calculate qs
                        qs = torch.tensor(
                            discount_reward(rewards, GAMMA, v_final_state), dtype=torch.float, device=DEVICE)

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
                            for _ in np.arange(N_TEST_EPI):

                                test_state = test_env.reset()
                                test_ep_reward = 0
                                test_done = False

                                # every test episdoe
                                while not test_done:

                                    # get action
                                    if is_cont:
                                        test_action_outputs = actor_network(torch.tensor([test_state], dtype=torch.float, device=DEVICE))
                                        mean, std = test_action_outputs[:, :1], test_action_outputs[:, 1:]
                                        std = std.abs()
                                        action_dist = Normal(mean, std + EPS)
                                        test_action = action_dist.sample().cpu().numpy()[0]
                                    else:
                                        test_softmax_action = torch.exp(
                                            actor_network(torch.tensor([test_state], dtype=torch.float, device=DEVICE)))
                                        test_action = np.argmax(
                                            test_softmax_action.cpu().numpy()[0])

                                    # take action
                                    test_next_state, test_reward, test_done, _ = test_env.step(
                                        test_action)

                                    # test_env.render()

                                    test_ep_reward += test_reward
                                    test_state = test_next_state
                                    if test_done:
                                        break

                                # collect rewards for this testing
                                test_rewards.append(test_ep_reward)

                            # collect rewards for every testing
                            total_test_rewards.append(test_rewards)

                            print("Step: {}, Average test rewards: {}".format(step + 1, np.mean(test_rewards)))

                np.save(result_file_name, np.array(total_test_rewards))
