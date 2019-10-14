import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict


def main():
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
    for file_ in os.listdir('results/'):
        if file_.endswith('.npy'):
            file_name = file_.split('_')
            if file_name[0] == 'CartPole-v0':
                CartPole[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))
            elif file_name[0] == 'MountainCar-v0':
                MountainCar[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))
            elif file_name[0] == 'Acrobot-v1':
                Acrobot[int(file_name[1])][int(file_name[2][:-4])] = np.load(os.path.join('results/', file_))

    for ENV, STEP_ROLLOUTS in zip(["CartPole-v0", "MountainCar-v0", 'Acrobot-v1'], [[1, 10, 40, 80, 150, 200], [1, 10, 40, 80, 150, 200], [1, 30, 60, 100, 300, 500]]):
        # ENV = 'CartPole-v0'
        # STEP_ROLLOUTS = [1, 10, 40, 80, 150, 200]
        for SEED in [42, 36, 15]:
            plt.figure(figsize=(10, 5))
            for STEP_ROLLOUT in STEP_ROLLOUTS:
                if ENV == 'CartPole-v0':
                    mean = CartPole[SEED][STEP_ROLLOUT].mean(axis=1)
                    std = CartPole[SEED][STEP_ROLLOUT].std(axis=1)
                elif ENV == 'MountainCar-v0':
                    mean = MountainCar[SEED][STEP_ROLLOUT].mean(axis=1)
                    std = MountainCar[SEED][STEP_ROLLOUT].std(axis=1)
                elif ENV == 'Acrobot-v1':
                    mean = Acrobot[SEED][STEP_ROLLOUT].mean(axis=1)
                    std = Acrobot[SEED][STEP_ROLLOUT].std(axis=1)

                plt.plot(mean)
                plt.fill_between(np.arange(100), mean - std, mean + std, alpha=0.3, label='n={}'.format(STEP_ROLLOUT))
            plt.xlabel('Step (x100)')
            plt.ylabel('Average Return')
            plt.title('{} Seed={}'.format(ENV, SEED))
            plt.legend()
            plt.show()


if __name__ == '__main__':
    main()
