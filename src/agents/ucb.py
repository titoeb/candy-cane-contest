#! pip install kaggle-environments --upgrade -q

import numpy as np
import random


class UpperConfidenceBound:
    def __init__(self, c, n_bins):
        self._c = c
        self._n_bins = n_bins
        self.initiallize(n_bins=n_bins, c=c)

    def initiallize(self, n_bins, c):
        self.c = c
        self.n_bins = n_bins
        self.n_used = np.full(n_bins, 1e-10)
        self.rewards = np.full(n_bins, 1e-10)
        self.total_reward = 0

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(n_bins=self._n_bins, c=self._c)
        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.rewards[my_last_action] += reward
            self.n_used[my_last_action] += 1
            self.total_reward = observation.reward

        # Choose action
        # Compute ucb target function
        success_ratio = self.rewards / self.n_used
        t = observation.step
        exploration = self.c * np.sqrt(np.log(t + 1) / self.n_used)

        return int(np.argmax(success_ratio + exploration))


upper_confidence_bound = UpperConfidenceBound(n_bins=100, c=0.45)


def agent(observation, configuration):
    return upper_confidence_bound.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
