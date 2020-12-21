import datetime
from typing import List, Callable
import numpy as np
import datetime
import typer
from utils import simulate_mab


# AGENTS = ["agent_ucb_new.py", "agent_baysian_ucb.py"]
# AGENTS = ["agent_vegas_slot_machines.py", "agent_ucb_new.py"]


def main(agent1: str, agent2: str, n_rounds: int, n_processes: int):
    AGENTS = [agent1, agent2]
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

    print("Agent \t\t Average Reward \t\t Standard deviation of reward")
    print("-" * 100)
    print(
        "\n".join(
            [
                f"{agent} \t\t {round(np.array(rewards).mean(), 3)} \t\t {round(np.std(np.array(rewards)), 3)}"
                for agent, rewards in zip(AGENTS, rewards)
            ]
        )
    )
    print("-" * 100)


if __name__ == "__main__":
    typer.run(main)
