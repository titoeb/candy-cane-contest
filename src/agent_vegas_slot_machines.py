#! pip install kaggle-environments --upgrade -q

import numpy as np
import random


class VegasSlotMachine:
    def __init__(self, c, n_bins):
        self._c = c
        self._n_bins = n_bins
        self.initiallize(n_bins=n_bins, c=c)

    def initiallize(self, n_bins, c):
        self.c = c
        self.n_bins = n_bins
        self.total_reward = 0

        self.success = np.ones(n_bins)
        self.failure = np.zeros(n_bins)
        self.opponent_pulls = np.zeros(n_bins)

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(n_bins=self._n_bins, c=self._c)
            return int(np.random.randint(0, self.n_bins))
        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            opponent_last_action = observation.lastActions[1 - my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.total_reward = observation.reward
            self.success[my_last_action] += reward
            self.failure[my_last_action] += 1 - reward
            self.opponent_pulls[opponent_last_action] += 1

            # Choose action
            # Compute ucb target function
            target_function = (
                self.success
                - self.failure
                + self.opponent_pulls
                - (self.opponent_pulls > 0) * self.c
            ) / (self.success + self.failure + self.opponent_pulls)

            return int(np.argmax(target_function))


vegas_slot_machine = VegasSlotMachine(n_bins=100, c=0.25)


def agent(observation, configuration):
    return vegas_slot_machine.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
