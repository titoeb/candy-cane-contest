from agents import (
    random_agent,
    cycle_agent,
    GreedyAgent,
    EpsilonGreedyAgent,
    EpsilonDecayingGreedyAgent,
    ThomsonSampler,
)
import datetime
from typing import List, Callable
import numpy as np
import datetime
import typer
from utils import simulate_mab


def main(n_rounds: int, n_processes: int):

    # greedy_agent = GreedyAgent(n_bins=100)
    # epsilon_greedy_agent = EpsilonGreedyAgent(n_bins=100, epsilon=0.5)
    epsilon_decaying_greedy_agent = EpsilonDecayingGreedyAgent(
        n_bins=100, epsilon_start=1.0, ratio_decay=0.9995, min_epsilon=0.4
    )

    thomson = ThomsonSampler(n_bins=100)

    agents = [
        random_agent,
        # cycle_agent,
        # greedy_agent.play,
        # epsilon_greedy_agent.play,
        epsilon_decaying_greedy_agent.play,
        # thomson.play,
    ]

    print(
        f"Simulating multi_armed_bandit with {n_rounds} rounds parallelized on {n_processes} processes."
    )
    start = datetime.datetime.now()
    rewards = simulate_mab(agents, n_rounds=n_rounds, n_processes=n_processes)
    print(rewards)
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
