import datetime
from typing import List, Callable
import numpy as np
import datetime
import typer
from utils import (
    simulate_mab,
    print_sep,
    get_agent_final_rewards,
    get_agent_wins,
    get_summary_matches,
)
from pathlib import Path
import collections


def main(
    agent_to_evaluate: str,
    n_rounds_per_agent: int,
    n_processes: int,
    #    base_path: str = "agents/",
    base_path: str,
):
    # Load all agents
    if not agent_to_evaluate.endswith(".py"):
        agent_to_evaluate = agent_to_evaluate + ".py"

    # Load all agents.
    all_agents = list(Path(base_path).absolute().iterdir())
    agent_path = [
        agent_path
        for agent_path in all_agents
        if agent_path.name.startswith(agent_to_evaluate)
    ]

    opponent_pool_paths = [
        agent_path
        for agent_path in all_agents
        if agent_to_evaluate not in agent_path.name
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

    start = datetime.datetime.now()
    all_simulations = {
        opponent_path.name: simulate_mab(
            [agent_path.as_posix(), opponent_path.as_posix()],
            n_rounds_per_agent,
            n_processes,
        )
        for opponent_path in opponent_pool_paths
    }

    print(
        f"The totals simulation took {datetime.datetime.now() - start}.\n{print_sep()}"
    )

    average_reward_agent = get_agent_final_rewards(0, all_simulations).mean()
    win_ratio_agent = get_agent_wins(0, all_simulations).mean()

    print(
        f"Agent: {agent_path.name}, total average_reward: {average_reward_agent}, total win ratio: {win_ratio_agent}\n{print_sep()}\n"
    )

    summary = get_summary_matches(0, all_simulations)
    summary_losses = {
        opponent: summary for (opponent, summary) in summary.items() if not summary[-1]
    }

    summary_wins = {
        opponent: summary for (opponent, summary) in summary.items() if summary[-1]
    }

    # Sort wins, losses by their win ratio.
    summary_losses_sorted = collections.OrderedDict(
        sorted(summary_losses.items(), key=lambda x: -x[1][1][0])
    )
    summary_wins_sorted = collections.OrderedDict(
        sorted(summary_wins.items(), key=lambda x: -x[1][1][0])
    )

    print(f"Agent {agent_path.name} has lost against:\n{print_sep()}\n")
    for agent, summary in summary_losses_sorted.items():
        print(f"Agent: {agent}\n{summary}\n")

    print(f"Agent {agent_path.name} has won against:\n{print_sep()}\n")
    for agent, summary in summary_wins_sorted.items():
        print(f"Agent: {agent}\n{summary}\n")

    print("Execution done.")

    # print("Agent \t\t Average Reward \t\t Standard deviation of reward")
    # print("-" * 100)
    # print(
    #     "\n".join(
    #         [
    #             f"{agent} \t\t {round(np.array(rewards).mean(), 3)} \t\t {round(np.std(np.array(rewards)), 3)}"
    #             for agent, rewards in zip(AGENTS, rewards)
    #         ]
    #     )
    # )
    # print("-" * 100)


if __name__ == "__main__":
    typer.run(main)
