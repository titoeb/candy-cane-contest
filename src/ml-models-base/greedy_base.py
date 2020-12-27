import numpy as np
import random
import pickle
import base64
from typing import Any


def decode_base64_string(data: str) -> Any:
    return pickle.loads(base64.b64decode(data))


class Greedy_ML:
    def __init__(self, n_bins: int, consider_outcomes: float, model_str: str):
        self.consider_outcomes = consider_outcomes
        self.n_bins = n_bins
        self.model = decode_base64_string(model_str)

        # Create data
        self.total_reward = 0
        self.n_pulls = np.full(self.n_bins, 0.0)
        self.success = np.full(self.n_bins, 0.0)
        self.n_pulls_opponent = np.full(self.n_bins, 0.0)

        # Create initial predictions
        # Currently all features are zeros in the beginning.
        self.predictions = self.model.predict(np.full((self.n_bins, 4), 0.0))

    def play(self, observation, configuration):
        my_index = observation.agentIndex
        # Extract info from observation.
        my_last_action = observation.lastActions[my_index]
        opponent_last_action = observation.lastActions[1 - my_index]
        reward = observation.reward - self.total_reward

        # Extract params
        self.n_pulls[my_index] += 1
        self.n_pulls_opponent[opponent_last_action] += 1
        self.success[my_index] += reward

        # Update predictions for chosen machines.
        # Update predictions for machine pulled by itself
        self.predictions[my_last_action] = self.model.predict(
            [
                self.n_pulls_opponent.sum(),
                self.n_pulls[my_last_action],
                self.success[my_last_action],
                self.n_pulls_opponent[my_last_action],
            ]
        )

        # Update predictions for machine opponent pulled.
        self.predictions[opponent_last_action] = self.model.predict(
            [
                self.n_pulls_opponent.sum(),
                self.n_pulls[opponent_last_action],
                self.success[opponent_last_action],
                self.n_pulls_opponent[opponent_last_action],
            ]
        )

        # Sample action from all actions that are within
        # self.consider_outcomes * maximum outcome
        return int(
            np.random.choice(
                np.where(
                    self.predictions > self.predictions.max() * self.consider_outcomes
                )[0]
            )
        )


greedy_ml = Greedy_ML(n_bins=100, ...)


def agent(observation, configuration):
    return greedy_ml.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
