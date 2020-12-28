import pandas as pd
import numpy as np
import pickle
from typing import Dict, List
import datetime


RESULT_DIR = "/usr/src/data/results_2020-12-28--14-18-54.pickle"


def create_training_data(
    result: List[Dict],
) -> pd.DataFrame:

    # Get number of bins and opponents from the data
    n_bins = len(result[0][0]["observation"]["thresholds"])

    # Create raw dataset.
    initial = np.full((n_bins + 2 * (len(result) - 1), 5), 0.0)
    data = pd.DataFrame(
        data=initial,
        columns=[
            "round",
            "n_pulls_self",
            "n_success_self",
            "n_pulls_opponent",
            "success_probs",
        ],
    )

    # Pre insert round and agent
    data["round"] = [0] * n_bins + sorted(list(range(1, len(result))) * 2)
    data["agent_id"] = [-1] * n_bins + [0, 1] * (len(result) - 1)

    # Set initial rewards
    data.iloc[:100].loc[:, "success_probs"] = [
        threshold / 100 for threshold in result[0][0]["observation"]["thresholds"]
    ]
    agent_0_state = {
        "n_pulls_self": np.full(n_bins, 0.0),
        "n_success_self": np.full(n_bins, 0.0),
        "n_pulls_opponent": np.full(n_bins, 0.0),
        "total_reward": 0,
    }

    agent_1_state = {
        "n_pulls_self": np.full(n_bins, 0.0),
        "n_success_self": np.full(n_bins, 0.0),
        "n_pulls_opponent": np.full(n_bins, 0.0),
        "total_reward": 0,
    }

    success_ratios = np.full(n_bins, 0.0)

    for round_agent_0, round_agent_1 in result:
        current_step = round_agent_0["observation"]["step"]
        if current_step > 0:
            # Update agents
            agent_0_action = round_agent_0["observation"]["lastActions"][
                round_agent_0["observation"]["agentIndex"]
            ]
            agent_1_action = round_agent_0["observation"]["lastActions"][
                round_agent_1["observation"]["agentIndex"]
            ]

            agent_0_current_reward = (
                round_agent_0["reward"] - agent_0_state["total_reward"]
            )
            agent_0_state["total_reward"] = round_agent_0["reward"]

            agent_1_current_reward = (
                round_agent_1["reward"] - agent_1_state["total_reward"]
            )
            agent_1_state["total_reward"] = round_agent_1["reward"]

            agent_0_state["n_pulls_self"][agent_0_action] += 1
            agent_0_state["n_success_self"][agent_0_action] += agent_0_current_reward
            agent_0_state["n_pulls_opponent"][agent_1_action] += 1

            agent_1_state["n_pulls_self"][agent_1_action] += 1
            agent_1_state["n_success_self"][agent_1_action] += agent_1_current_reward
            agent_1_state["n_pulls_opponent"][agent_0_action] += 1

            # Insert one datapoint for each agent
            idx_agent_0 = n_bins + (current_step - 1) * 2
            data.iloc[idx_agent_0, :].loc[
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_0_state["n_pulls_self"][agent_0_action],
                agent_0_state["n_success_self"][agent_0_action],
                agent_0_state["n_pulls_opponent"][agent_1_action],
                success_ratios[agent_0_action],
            ]

            # Insert one datapoint for each agent
            idx_agent_1 = n_bins + (current_step - 1) * 2 + 1
            data.iloc[idx_agent_1, :].loc[
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_1_state["n_pulls_self"][agent_1_action],
                agent_1_state["n_success_self"][agent_1_action],
                agent_1_state["n_pulls_opponent"][agent_0_action],
                success_ratios[agent_1_action],
            ]

        # Update rewards
        success_ratios = [
            threshold / 100 for threshold in round_agent_0["observation"]["thresholds"]
        ]

    return data


if __name__ == "__main__":

    with open(RESULT_DIR, "rb") as file_handler:
        results = pickle.load(file_handler)

    df = pd.concat(
        [create_training_data(result=results[0]) for results in results.values()]
    )

    file_name = (
        f'data/data_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.parquet'
    )
    df.to_parquet(file_name)
