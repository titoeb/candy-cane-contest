import typer
from evaluate import evaluate_against_opponents
from utils import print_sep
from pathlib import Path


def main(
    candidate: str,
    opponent: str,
    n_rounds_per_agent: int,
    n_processes: int,
    base_path: str,
):
    # Load all agents
    if not candidate.endswith(".py"):
        candidate = candidate + ".py"

    if not opponent.endswith(".py"):
        opponent = opponent + ".py"

    all_agents = list(Path(base_path).absolute().iterdir())

    agent_path = [
        agent_path for agent_path in all_agents if agent_path.name.startswith(candidate)
    ].pop()
    opponent_path = [
        agent_path for agent_path in all_agents if agent_path.name.startswith(opponent)
    ]

    print(print_sep())
    print(
        f"The agent {agent_path.name} will be tested against the following agent: {opponent_path[0].name}\n"
        f"The candidate agent will be tested for {n_rounds_per_agent} rounds against each agent. \nThe process parallelized on {n_processes} processes."
    )

    prints, _ = evaluate_against_opponents(
        agent_path, opponent_path, n_rounds_per_agent, n_processes
    )
    print(prints)


if __name__ == "__main__":
    typer.run(main)