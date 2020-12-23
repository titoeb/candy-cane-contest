#! pip install kaggle-environments --upgrade -q

import numpy as np
import random


class Boltzman:
    def __init__(self, n_bins, invT):
        self._n_bins = n_bins
        self._invT = invT
        self.initiallize(n_bins=n_bins, invT=invT)

    def initiallize(self, n_bins, invT):
        self.n_bins = n_bins
        self.invT = invT

        self.total_reward = 0
        self.success = np.ones(n_bins)
        self.failure = np.zeros(n_bins)

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(n_bins=self._n_bins, invT=self.invT)
            return int(np.random.randint(0, self.n_bins))
        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.total_reward = observation.reward
            self.success[my_last_action] += reward
            self.failure[my_last_action] += 1 - reward

            # Choose action
            success_ratio = self.success / (self.success + self.failure)
            weight = np.exp(self.invT * success_ratio)
            return int(np.random.choice(self.n_bins, 1, p=list(weight / weight.sum())))


boltzman = Boltzman(n_bins=100, invT=10)


def agent(observation, configuration):
    return boltzman.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
