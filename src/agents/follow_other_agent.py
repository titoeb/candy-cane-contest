import numpy as np
import random


def agent(observation, configuration):
    """
    Very simple agent that just follows the actions of the other agent.
    """
    if observation.step == 0:
        return 0
    else:
        return observation.lastActions[1 - observation.agentIndex]


# if __name__ == "__main__":
#     from kaggle_environments import make

#     def random_agent(observation, configuration):
#         return random.randrange(configuration.banditCount)

#     env = make("mab", debug=True)

#     # Run environment
#     steps = env.run([random_agent, agent])
#     print(steps[-1][0].reward, steps[-1][1].reward)
