#! pip install kaggle-environments --upgrade -q

import numpy as np
import random
import collections


class UpperConfidenceBound:
    def __init__(self, c, n_bins, buffer_length, damping_factor):
        self._c = c
        self._buffer_length = buffer_length
        self._damping_factor = damping_factor
        self._n_bins = n_bins
        self.initiallize(
            n_bins=n_bins,
            c=c,
            buffer_length=buffer_length,
            damping_factor=damping_factor,
        )

    def initiallize(self, n_bins, c, buffer_length, damping_factor):
        self.damping_factor = damping_factor
        self.c = c
        self.n_bins = n_bins

        self.opponent_action_buffer = collections.deque(maxlen=buffer_length)
        self.n_used = np.full(n_bins, 1e-10)
        self.rewards = np.full(n_bins, 1e-10)
        self.total_reward = 0

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(
                n_bins=self._n_bins,
                c=self._c,
                buffer_length=self._buffer_length,
                damping_factor=self._damping_factor,
            )
        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            opponent_last_action = observation.lastActions[1 - my_index]
            self.opponent_action_buffer.append(opponent_last_action)

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

        target_function = success_ratio + exploration

        # If the opponent used the action recently (e.g. it is in the buffer)
        # incraese the target function multiplicativly.
        recent_actions = set(self.opponent_action_buffer)
        values = (
            1
            + np.log(
                1
                + [
                    self.opponent_action_buffer.count(recent_action)
                    for recent_action in recent_actions
                ]
            )
            / self.damping_factor
        )

        factor = np.ones(self.n_bins)
        factors[list(recent_actions)] *= values

        return int(np.argmax(target_function * factors))


upper_confidence_bound = UpperConfidenceBound(
    n_bins=100, c=0.45, buffer_length=100, damping_factor=2
)


def agent(observation, configuration):
    return upper_confidence_bound.play(
        observation,
        configuration,
    )


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
