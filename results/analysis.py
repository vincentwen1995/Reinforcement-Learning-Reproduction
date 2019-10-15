import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict


def analyze():
    # steps = []
    # return_means = []
    # return_stds = []
    # with open('results/MC_base.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     for row in csv_reader:
    #         steps.append(row[0])
    #         return_means.append(row[1])
    #         return_stds.append(row[2])

    # steps = np.array(steps)
    # return_means = np.array(return_means, dtype=np.float)
    # return_stds = np.array(return_stds, dtype=np.float)

    # plt.figure(figsize=(10, 5))
    # plt.plot(return_means)
    # plt.fill_between(steps, return_means - return_stds, return_means + return_stds, alpha=0.5)
    # plt.show()
    CartPole = defaultdict(lambda: dict())
    MountainCar = defaultdict(lambda: dict())
    Acrobot = defaultdict(lambda: dict())
    Pendulum = defaultdict(lambda: dict())

    for file_ in os.listdir('results/'):
        if file_.endswith('.npy'):
            file_name = file_.split('_')
            if file_name[0] == 'CartPole-v0':
                CartPole[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))
            elif file_name[0] == 'MountainCar-v0':
                MountainCar[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))
            elif file_name[0] == 'Acrobot-v1':
                Acrobot[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))
            elif file_name[0] == 'Pendulum-v0':
                Pendulum[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))

    for env, step_rollouts in \
            zip(["CartPole-v0", "MountainCar-v0", 'Acrobot-v1', 'Pendulum-v0'], [[1, 10, 40, 80, 150, 200], [1, 10, 40, 80, 150, 200], [1, 30, 60, 100, 300, 500], [1, 10, 40, 80, 150, 200]]):
        # env = 'CartPole-v0'
        # step_rollouts = [1, 10, 40, 80, 150, 200]
        plt.figure(figsize=(10, 5))
        # for seed in [42, 36, 15]:
        for step_rollout in step_rollouts:
            rewards = []
            # for step_rollout in step_rollouts:
            for seed in [42, 36, 15]:
                if env == 'CartPole-v0':
                    rewards.append(CartPole[seed][step_rollout])
                    # mean = CartPole[seed][step_rollout].mean(axis=1)
                    # std = CartPole[seed][step_rollout].std(axis=1)
                elif env == 'MountainCar-v0':
                    rewards.append(MountainCar[seed][step_rollout])
                    # mean = MountainCar[seed][step_rollout].mean(axis=1)
                    # std = MountainCar[seed][step_rollout].std(axis=1)
                elif env == 'Acrobot-v1':
                    rewards.append(Acrobot[seed][step_rollout])
                    # mean = Acrobot[seed][step_rollout].mean(axis=1)
                    # std = Acrobot[seed][step_rollout].std(axis=1)
                elif env == 'Pendulum-v0':
                    rewards.append(Pendulum[seed][step_rollout])
            rewards = np.hstack(rewards)
            mean = rewards.mean(axis=1)
            std = rewards.std(axis=1)
            # plt.errorbar(np.arange(100), mean, yerr=std, solid_capstyle='projecting', capsize=5, label='n={}'.format(step_rollout))
            plt.plot(mean)
            plt.fill_between(np.arange(100), mean - std, mean + std, alpha=0.3, label='n={}'.format(step_rollout))
        plt.xlabel('Step (x100)')
        plt.ylabel('Average Return')
        # plt.title('{} Seed={}'.format(env, seed))
        plt.title('{}'.format(env))
        plt.legend()
        plt.show()
