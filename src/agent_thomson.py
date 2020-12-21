import numpy as np
import random


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
            self.posterior_a[my_last_action] += reward
            self.posterior_b[my_last_action] += 1 - reward

        samples = np.random.beta(a=self.posterior_a, b=self.posterior_b)
        return int(np.argmax(samples))


thompson_sampler = ThomsonSampler(n_bins=100)


def agent(observation, configuration):
    return thompson_sampler.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
