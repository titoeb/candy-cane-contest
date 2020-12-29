import datetime
from typing import List, Callable, Tuple, Dict
import numpy as np
import typer
from utils import (
    simulate_mab,
    print_sep,
    get_agent_final_rewards,
    get_agent_wins,
    get_summary_matches,
    my_tqdm,
)
from pathlib import Path
import collections
from tqdm import tqdm
import pickle
from evaluate import evaluate_against_opponents


def main(
    agent_to_evaluate: str,
    opponents: List[str],
    n_rounds_per_agent: int,
    n_processes: int,
    #    base_path: str = "agents/",
    base_path: str,
    store_results: bool = True,
):
    # Load all agents
    if not agent_to_evaluate.endswith(".py"):
        agent_to_evaluate = agent_to_evaluate + ".py"

    opponents = [
        opponent + ".py" if not opponent.endswith(".py") else opponent
        for opponent in opponents
    ]

    # Load all opponents
    all_agents = list(Path(base_path).absolute().iterdir())
    agent_path = [
        agent_path
        for agent_path in all_agents
        if agent_path.name.startswith(agent_to_evaluate)
        and not agent_path.name == "__pycache__"
    ]

    opponent_pool_paths = [
        agent_path
        for agent_path in all_agents
        if any(agent_path.name.startswith(opponent) for opponent in opponents)
        and not agent_path.name == "__pycache__"
    ]

    if len(agent_path) > 1:
        raise ValueError(
            f"You specified the following agent to be evaluated {agent_to_evaluate}, but there are more then one candidate: {', '.join([agent.name for agent in agent_path])}"
        )
    else:
        agent_path = agent_path.pop()

    print(print_sep())
    print(
        f"The agent {agent_path.name} will be tested against the following agents: {', '.join([agent.name for agent in opponent_pool_paths])}\n"
        f"The candidate agent will be tested for {n_rounds_per_agent} rounds against each agent. \nThe process parallelized on {n_processes} processes."
    )

    prints, results = evaluate_against_opponents(
        agent_path, opponent_pool_paths, n_rounds_per_agent, n_processes
    )
    print(prints)
    if store_results is True:
        # Store results in data
        file_name = f'data/results_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pickle'
        with open(file_name, "wb") as file_handler:
            pickle.dump(results, file_handler)

        print(f"Store results to {file_name}")


if __name__ == "__main__":
    typer.run(main)
