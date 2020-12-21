import numpy as np
import random


def cycle_agent(observation, configuration):
    return observation["step"] % configuration.banditCount


def agent(observation, configuration):
    return cycle_agent


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
