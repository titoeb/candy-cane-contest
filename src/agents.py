import random
import numpy as np
import random

# Create a test-agent that chooses one action randomly:
def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)


# Create a test-agent that cycles through the possible actions:
def cycle_agent(observation, configuration):
    return observation["step"] % configuration.banditCount


# Create thomson-sampling:
class ThomsonSamplingAgent:
    """
    I am using a class here to save the state of the agent.
    """

    def __init__(self, n_bins):
        self.n_called = 0

    def play(self, observation, configuration):
        self.n_called += 1
        return cycle_agent(observation, configuration)


class EpsilonDecayingGreedyAgent:
    def __init__(
        self, n_bins: int, epsilon_start: float, ratio_decay: float, min_epsilon: float
    ):
        self.epsilon_greedy_agent = EpsilonGreedyAgent(
            n_bins=n_bins, epsilon=epsilon_start
        )
        self.ratio_decay = ratio_decay
        self.min_epsilon = min_epsilon

    def play(self, observation, configuration):

        action = self.epsilon_greedy_agent.play(observation, configuration)

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


class ThomsonSampler:
    def __init__(self, n_bins):
        self.initiallize(n_bins)

    def initiallize(self, n_bins):
        self.n_bins = n_bins
        self.posterior_a = np.ones(n_bins)
        self.posterior_b = np.ones(n_bins)
        self.total_reward = 0

    def play(self, observation, configuration):

        my_index = observation.agentIndex
        if observation.step == 0:
            self.initiallize(n_bins=self.n_bins)

        else:
            my_last_action = observation.lastActions[my_index]
            reward = observation.reward - self.total_reward
            self.total_reward = observation.reward
            if reward != 0 and reward != 1:
                print("hi1")

            self.posterior_a[my_last_action] += reward
            self.posterior_b[my_last_action] += 1 - reward

        samples = np.random.beta(self.posterior_b, self.posterior_a)
        return int(np.argmax(samples))


class EpsilonGreedyAgent:
    def __init__(self, n_bins: int, epsilon: float):
        self.greedy_agent = GreedyAgent(n_bins=n_bins)
        self.epsilon = epsilon

    def play(self, observation, configuration):
        greedy_action = self.greedy_agent.play(observation, configuration)
        if random.random() > self.epsilon:
            # print(f"Took greedy action {greedy_action}")
            return greedy_action
        else:
            random_action = random_agent(observation, configuration)
            # print(f"Took random action {random_action}")
            return random_action


class GreedyAgent:
    def __init__(self, n_bins: int):
        self.success = np.zeros(n_bins)
        self.failure = np.zeros(n_bins)
        self.reward = 0

    def play(self, observation, configuration):

        # Update last action.
        if len(observation["lastActions"]) > 0:
            # If not any previous action stored, this is the first round
            # we don't need to update our parameters.
            last_action = observation["lastActions"][observation["agentIndex"]]

            # If the stored reword is not the reward in observation,
            # the last action was a success, otherwise a failure.
            if self.reward != observation["reward"]:
                self.success[last_action] += 1
            else:
                self.failure[last_action] += 1
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