from typing import List, Callable
import multiprocessing as mp
from kaggle_environments import make, evaluate
from copy import deepcopy


def play_mab(agents: List[str]) -> List[int]:
    """
    Play a round of multi-armed bandit with the specified agents. Return
    a list of the rewards of the different agents.
    """

    # Create environment
    env = make("mab", debug=True)

    # Run environment
    steps = env.run(agents)

    # Compute rewards
    final_step = steps[-1]
    return [final_step[0].reward, final_step[1].reward]


def simulate_mab(agents: List[str], n_rounds: int, n_processes: int) -> List[List[int]]:
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

    if n_processes > 1 and n_rounds > 1:
        pool = mp.Pool(n_processes)
        rewards = pool.starmap(
            play_mab,
            [(agents,) for _ in range(n_rounds)],
        )
        pool.close()
    else:
        # No multiprocessing
        rewards = [play_mab(agents) for _ in range(n_rounds)]

    return [elem for elem in zip(*rewards)]
