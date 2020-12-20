from kaggle_environments import make, evaluate
from random_agent import random_agent, cycle_agent, thomson_sampling
import datetime
from typing import List, Callable
import multiprocessing as mp
import numpy as np
import datetime
import typer


def play_mab(agents: List[Callable], print_results: bool) -> List[int]:
    """
    Play a round of multi-armed bandit with the specified agents. Return
    a list of the rewards of the different agents.
    """

    # Create environment
    env = make("mab", debug=True)

    # Run environment
    steps = env.run(agents)

    # Compute rewards
    return evaluate("mab", agents, steps)


def simulate_mab(
    agents: List[Callable], n_rounds: int, n_processes: int
) -> List[List[int]]:
    """
    Play multiple rounds of mab and return the single outcomes.
    It returns a list of the following structure:
    [
        [reward-round-1-agent-1, reward-round-2-agent-1, ..., reward-round-n-agent-1],
        [reward-round-1-agent-2, reward-round-2-agent-2, ..., reward-round-n-agent-2],
        ...,
        [reward-round-1-agent-t, reward-round-2-agent-t, ..., reward-round-n-agent-t],
    ]
    """

    if n_processes > 1:
        pool = mp.Pool(n_processes)
        rewards = pool.starmap(play_mab, [(agents, False) for _ in range(n_rounds)])
        pool.close()
    else:
        # No multiprocessing
        rewards = [play_mab(agents, False) for _ in range(n_rounds)]

    return [elem for elem in zip(*[elem[0] for elem in rewards])]


def main(n_rounds: int, n_processes: int):

    agents = [random_agent, cycle_agent]

    print(
        f"Simulating multi_armed_bandit with {n_rounds} rounds parallelized on {n_processes} processes."
    )
    start = datetime.datetime.now()
    rewards = simulate_mab(agents, n_rounds=n_rounds, n_processes=n_processes)
    print(f"Execution took {datetime.datetime.now() - start}")

    print(
        "\n".join(
            [
                f"agent {agent.__name__} had an average reward of {round(np.array(rewards).mean(), 3)} with a standard devation of {round(np.std(np.array(rewards)), 3)}"
                for agent, rewards in zip(agents, rewards)
            ]
        )
    )


if __name__ == "__main__":
    typer.run(main)
