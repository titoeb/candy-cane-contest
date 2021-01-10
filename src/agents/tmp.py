RANDOM_TRESHOLD = 3.63482751867761
C = 0.014870133675403019

import numpy as np
import random


class VegasSlotMachine:
    def __init__(self, c, random_treshold, n_bins):
        self._c = c
        self._n_bins = n_bins
        self._random_threshold = random_treshold
        self.initiallize(n_bins=n_bins, c=c, random_treshold=random_treshold)

    def initiallize(self, n_bins, c, random_treshold):
        self.c = c
        self.random_threshold = random_treshold
        self.n_bins = n_bins
        self.total_reward = 0

        self.success = np.ones(n_bins)
        self.failure = np.zeros(n_bins)
        self.opponent_pulls = np.zeros(n_bins)
        self.opponent_again = np.zeros(n_bins)
        self.opponent_last_action = -1
        self.action_buffer = []

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(
                n_bins=self._n_bins, c=self._c, random_treshold=self._random_threshold
            )
            return int(np.random.randint(0, self.n_bins))
        elif observation.step == 1:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            opponent_last_action = observation.lastActions[1 - my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.total_reward = observation.reward
            self.success[my_last_action] += reward
            self.failure[my_last_action] += 1 - reward
            self.opponent_pulls[opponent_last_action] += 1
            self.opponent_last_action = opponent_last_action
            self.action_buffer.append(my_last_action)

        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            opponent_last_action = observation.lastActions[1 - my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.total_reward = observation.reward
            self.success[my_last_action] += reward
            self.failure[my_last_action] += 1 - reward
            if opponent_last_action == self.opponent_last_action:
                self.opponent_again[opponent_last_action] += 1
            self.opponent_last_action = opponent_last_action
            self.opponent_pulls[opponent_last_action] += 1
            self.action_buffer.append(my_last_action)

        # Choose action
        # Compute ucb target function
        target_function = (
            (
                self.success
                - self.failure
                + self.opponent_pulls
                - (self.opponent_pulls > 0) * self.c
                + self.opponent_again
            )
            / (self.success + self.failure + self.opponent_pulls)
            * np.power(0.97, self.success + self.failure + self.opponent_pulls)
        )

        if reward > 0 or (
            random.random() > self.random_threshold
            and observation.step > 3
            and self.action_buffer[-1] == self.action_buffer[-2]
            and self.action_buffer[-1] == self.action_buffer[-3]
        ):
            return my_last_action
        else:
            return int(np.argmax(target_function))


vegas_slot_machine = VegasSlotMachine(n_bins=100, c=C, random_treshold=RANDOM_TRESHOLD)


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
