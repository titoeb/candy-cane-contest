EXPLORATION = 3.1845722509543087e-06
SAMPLING = False
DECAYING = 1.172078863402782
DAMPING_FACTOR = None
BUFFER_LENGTH = 0
MIN_STEP_OPPONENT = 0

import numpy as np
import random
import collections


class UpperConfidenceBound:
    def __init__(
        self,
        c,
        n_bins,
        sampling,
        decaying,
        buffer_length,
        damping_factor,
        min_step_opponent,
    ):
        self._c = c
        self._n_bins = n_bins
        self._sampling = sampling
        self._decaying = decaying
        self._buffer_length = buffer_length
        self._damping_factor = damping_factor
        self._min_step_opponent = min_step_opponent
        self.initiallize(
            n_bins=n_bins,
            c=c,
            sampling=sampling,
            decaying=decaying,
            buffer_length=buffer_length,
            damping_factor=damping_factor,
            min_step_opponent=min_step_opponent,
        )

    def initiallize(
        self,
        n_bins,
        c,
        sampling,
        decaying,
        buffer_length,
        damping_factor,
        min_step_opponent,
    ):
        self.c = c
        self.decaying = decaying
        self.sampling = sampling
        self.buffer_length = buffer_length
        self.damping_factor = damping_factor
        self.n_bins = n_bins
        self.min_step_opponent = min_step_opponent

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
                sampling=self._sampling,
                decaying=self._decaying,
                buffer_length=self._buffer_length,
                damping_factor=self._damping_factor,
                min_step_opponent=self._min_step_opponent,
            )
        else:
            # Extract info from observation.
            my_last_action = observation.lastActions[my_index]
            opponent_last_action = observation.lastActions[1 - my_index]
            reward = observation.reward - self.total_reward

            # Extract params
            self.opponent_action_buffer.append(opponent_last_action)
            self.rewards[my_last_action] += reward
            self.n_used[my_last_action] += 1
            self.total_reward = observation.reward

            if self.decaying:
                self.rewards[my_last_action] *= self.decaying
                self.rewards[opponent_last_action] *= self.decaying

        # Choose action
        # Compute ucb target function
        success_ratio = self.rewards / self.n_used
        t = observation.step
        exploration = self.c * np.sqrt(np.log(t + 1) / self.n_used)
        target_function = success_ratio + exploration

        # If the opponent used the action recently (e.g. it is in the buffer)
        # incraese the target function multiplicativly.
        if self.damping_factor and t > self.min_step_opponent:
            recent_actions = set(self.opponent_action_buffer)
            values = (
                1
                + np.log(
                    [
                        1 + self.opponent_action_buffer.count(recent_action)
                        for recent_action in recent_actions
                    ]
                )
                / self.damping_factor
            )

            factors = np.ones(self.n_bins)
            factors[list(recent_actions)] *= values

            target_function = target_function * factors

        optimal_action = np.argmax(target_function)

        action_is_optimal_mask = target_function == target_function[optimal_action]
        if action_is_optimal_mask.sum() > 1 and self.sampling:
            # Sample a random one from the optimal actions
            return int(np.random.choice(np.where(action_is_optimal_mask)[0]))

        return int(optimal_action)


upper_confidence_bound = UpperConfidenceBound(
    n_bins=100,
    c=EXPLORATION,
    sampling=SAMPLING,
    decaying=DECAYING,
    buffer_length=BUFFER_LENGTH,
    damping_factor=DAMPING_FACTOR,
    min_step_opponent=MIN_STEP_OPPONENT,
)


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
