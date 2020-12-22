#! pip install kaggle-environments --upgrade -q

import numpy as np
import random
from scipy.stats import beta


class BaysianUpperConfidenceBound:
    def __init__(self, c, n_bins):
        self._c = c
        self._n_bins = n_bins
        self.initiallize(n_bins=n_bins, c=c)

    def initiallize(self, n_bins, c):
        self.c = c
        self.n_bins = n_bins
        self.posterior_a = np.ones(n_bins)
        self.posterior_b = np.ones(n_bins)
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
            self.posterior_a[my_last_action] += reward
            self.posterior_b[my_last_action] += 1 - reward
            self.total_reward = observation.reward

        # Choose action
        # Compute ucb target function
        upper_bound = (
            self.posterior_a / (self.posterior_a + self.posterior_b)
            + beta.std(self.posterior_a, self.posterior_b) * self.c
        )

        return int(np.argmax(upper_bound))


baysian_upper_confidence_bound = BaysianUpperConfidenceBound(n_bins=100, c=2.75)


def agent(observation, configuration):
    return baysian_upper_confidence_bound.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
