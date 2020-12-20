from kaggle_environments import make, evaluate
from random_agent import random_agent, cycle_agent, thomson_sampling
import datetime
from typing import List, Dict, Callable


def play_mab(agents: List[Callable], print_results: bool) -> Dict[str, int]:
    """
    Play a round of multi-armed bandit with the specified agents. Return
    a dictionary of agent_name -> agent_reward. The agent name is taken
    from agent.__name__.
    """

    # Create environment
    env = make("mab", debug=True)

    # Run environment
    steps = env.run(agents)

    # Compute rewards
    rewards = evaluate("mab", agents, steps)
    if print_results:
        print(
            "\n".join(
                [
                    f"agent {agent.__name__} got a reward of {reward}"
                    for agent, reward in zip(agents, rewards[0])
                ]
            )
        )
    return {}
