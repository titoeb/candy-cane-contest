import random

# Create a test-agent that chooses one action randomly:
def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)


# Create a test-agent that cycles through the possible actions:
def cycle_agent(observation, configuration):
    return observation["step"] % configuration.banditCount


# Create thomson-sampling:
def thomson_sampling(observation, configuration):
    print("hi!")
    return 1


# Only export agent
__all__ = ["random_agent"]