#! pip install kaggle-environments --upgrade -q

import numpy as np
import random


class BaseAgent:
    def __init__(self):
        self.mu = None
        self.win = None
        self.loss = None
        self.id = None
        self.tot_reward = 0

    def set_pars(self, obs, conf):
        self.n_bandits = conf.banditCount
        self.win = np.ones(self.n_bandits)
        self.loss = np.ones(self.n_bandits)
        self.id = obs.agentIndex

    def pull(self, obs, config):
        """random elf"""
        return int(np.random.choice(config.banditCount))


class Boltzman(BaseAgent):
    def __init__(self, invT):
        super(Boltzman, self).__init__()
        self.invT = invT

    def play(self, obs, conf):
        """boltzmann elf"""
        if obs.step == 0:
            self.set_pars(obs, conf)
        else:
            r = obs.reward - self.tot_reward
            self.tot_reward = obs.reward

            self.win[obs["lastActions"][self.id]] += r
            self.loss[obs["lastActions"][self.id]] += 1 - r

        mu = self.win / (self.win + self.loss)
        w = np.exp(self.invT * mu)

        return int(np.random.choice(self.n_bandits, 1, p=list(w / w.sum())))


boltzman = Boltzman(invT=10)


def agent(observation, configuration):
    return boltzman.play(observation, configuration)


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
