import datetime
from typing import List, Callable
import numpy as np
import datetime
import typer
from utils import simulate_mab


AGENTS = ["agent_ucb_new.py", "agent_thomson.py"]


def main(n_rounds: int, n_processes: int):

    print(
        f"Simulating multi_armed_bandit with {n_rounds} rounds parallelized on {n_processes} processes."
    )
    start = datetime.datetime.now()

    rewards = simulate_mab(
        [f"/usr/src/{agent}" for agent in AGENTS],
        n_rounds=n_rounds,
        n_processes=n_processes,
    )
    print(f"Execution took {datetime.datetime.now() - start}")

    print(
        "\n".join(
            [
                f"agent {agent} had an average reward of {round(np.array(rewards).mean(), 3)} with a standard devation of {round(np.std(np.array(rewards)), 3)}"
                for agent, rewards in zip(AGENTS, rewards)
            ]
        )
    )


if __name__ == "__main__":
    typer.run(main)
