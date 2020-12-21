import numpy as np
import random


def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)


class GreedyAgent:
    def __init__(self, n_bins: int):
        self._nbins = n_bins
        self.initiallize(n_bins=n_bins)

    def initiallize(self, n_bins):
        self.success = np.zeros(n_bins)
        self.failure = np.zeros(n_bins)
        self.reward = 0

    def play(self, observation, configuration):

        if observation.step == 0:
            self.initiallize(n_bins=self._nbins)
        # Update last action.
        if len(observation["lastActions"]) > 0:
            # If not any previous action stored, this is the first round
            # we don't need to update our parameters.
            last_action = observation["lastActions"][observation["agentIndex"]]

            # If the stored reword is not the reward in observation,
            # the last action was a success, otherwise a failure.
            reward = observation["reward"] - self.reward
            self.success[last_action] += reward
            self.failure[last_action] += 1 - reward
            self.reward = observation["reward"]

        # Compute new action
        success_ratio = np.round(
            self.success / (self.success + self.failure + 1e-10), decimals=4
        )
        optimal_action = np.argmax(success_ratio)
        action_is_optimal_mask = success_ratio == success_ratio[optimal_action]
        if action_is_optimal_mask.sum() > 1:
            # There are more than one lever with highest success ratio, sample from the remaining ones.
            # Typcast to int neccessary because of framework.
            return int(np.random.choice(np.where(action_is_optimal_mask)[0]))
        else:
            return int(optimal_action)


class EpsilonGreedyAgent:
    def __init__(self, n_bins: int, epsilon: float):
        self._epsilon = epsilon
        self._n_bins = n_bins
        self.initiallize(n_bins=self._n_bins, epsilon=self._epsilon)

    def initiallize(self, n_bins, epsilon):
        self.greedy_agent = GreedyAgent(n_bins=n_bins)
        self.epsilon = epsilon

    def play(self, observation, configuration):

        # Reset agent if new game started
        if observation.step == 0:
            self.initiallize(n_bins=self._n_bins, epsilon=self._epsilon)

        greedy_action = self.greedy_agent.play(observation, configuration)
        if random.random() > self.epsilon:
            # print(f"Took greedy action {greedy_action}")
            return greedy_action
        else:
            random_action = random_agent(observation, configuration)
            # print(f"Took random action {random_action}")
            return random_action


class EpsilonDecayingGreedyAgent:
    def __init__(
        self, n_bins: int, epsilon_start: float, ratio_decay: float, min_epsilon: float
    ):
        self._nbins = n_bins
        self._epsilon_start = epsilon_start
        self._ratio_decay = ratio_decay
        self._min_epsilon = min_epsilon

        self.initiallize(
            n_bins=self._nbins,
            epsilon_start=self._epsilon_start,
            ratio_decay=self._ratio_decay,
            min_epsilon=self._min_epsilon,
        )

    def initiallize(self, n_bins, epsilon_start, ratio_decay, min_epsilon):
        self.epsilon_greedy_agent = EpsilonGreedyAgent(
            n_bins=n_bins, epsilon=epsilon_start
        )
        self.ratio_decay = ratio_decay
        self.min_epsilon = min_epsilon

    def play(self, observation, configuration):

        action = self.epsilon_greedy_agent.play(observation, configuration)

        if observation.step == 0:
            self.initiallize(
                n_bins=self._nbins,
                epsilon_start=self._epsilon_start,
                ratio_decay=self._ratio_decay,
                min_epsilon=self._min_epsilon,
            )

        # Update epsilon
        self.epsilon_greedy_agent.epsilon = max(
            self.epsilon_greedy_agent.epsilon * self.ratio_decay, self.min_epsilon
        )

        # print(self.epsilon_greedy_agent.epsilon)
        # print(
        #     (
        #         self.epsilon_greedy_agent.greedy_agent.success
        #         + self.epsilon_greedy_agent.greedy_agent.success
        #     ).sum()
        # )

        return action


epsilon_decaying_greedy_agent = EpsilonDecayingGreedyAgent(
    n_bins=100, epsilon_start=1.0, ratio_decay=0.995, min_epsilon=0.6
)


def agent(observation, configuration):
    return epsilon_decaying_greedy_agent.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
