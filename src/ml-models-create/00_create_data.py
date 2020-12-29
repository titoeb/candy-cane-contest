import pandas as pd
import numpy as np
import pickle
from typing import Dict, List
import datetime
from tqdm import tqdm

RESULT_DIR = "/usr/src/data/results_2020-12-29--13-36-29.pickle"


def create_training_data(
    result: List[Dict],
) -> pd.DataFrame:

    # Get number of bins and opponents from the data
    n_bins = len(result[0][0]["observation"]["thresholds"])

    # Create raw dataset.
    initial = np.full((n_bins + 4 * (len(result) - 1), 5), 0.0)
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
    data["round"] = [0] * n_bins + sorted(list(range(1, len(result))) * 4)
    data["agent_id"] = [-1] * n_bins + [0, 1, 0, 1] * (len(result) - 1)

    # Set initial rewards
    data.loc[:99, "success_probs"] = [
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

            # Add 2 datapoints for each agents
            row = n_bins + (current_step - 1) * 4

            # Action agent 0 from perspective of agent 0
            data.at[row, "n_pulls_self"] = agent_0_state["n_pulls_self"][agent_0_action]
            data.at[row, "n_success_self"] = agent_0_state["n_success_self"][
                agent_0_action
            ]
            data.at[row, "n_pulls_opponent"] = agent_0_state["n_pulls_opponent"][
                agent_0_action
            ]
            data.at[row, "success_probs"] = success_ratios[agent_0_action]

            # Action agent 0 from perspective of agent 1
            data.at[row + 1, "n_pulls_self"] = agent_1_state["n_pulls_self"][
                agent_0_action
            ]
            data.at[row + 1, "n_success_self"] = agent_1_state["n_success_self"][
                agent_0_action
            ]
            data.at[row + 1, "n_pulls_opponent"] = agent_1_state["n_pulls_opponent"][
                agent_0_action
            ]
            data.at[row + 1, "success_probs"] = success_ratios[agent_0_action]

            # Action agent 1 from perspective of agent 1
            data.at[row + 2, "n_pulls_self"] = agent_1_state["n_pulls_self"][
                agent_1_action
            ]
            data.at[row + 2, "n_success_self"] = agent_1_state["n_success_self"][
                agent_1_action
            ]
            data.at[row + 2, "n_pulls_opponent"] = agent_1_state["n_pulls_opponent"][
                agent_1_action
            ]
            data.at[row + 2, "success_probs"] = success_ratios[agent_1_action]

            # Action agent 1 from perspective of agent 0
            data.at[row + 3, "n_pulls_self"] = agent_0_state["n_pulls_self"][
                agent_1_action
            ]
            data.at[row + 3, "n_success_self"] = agent_0_state["n_success_self"][
                agent_1_action
            ]
            data.at[row + 3, "n_pulls_opponent"] = agent_0_state["n_pulls_opponent"][
                agent_1_action
            ]
            data.at[row + 3, "success_probs"] = success_ratios[agent_1_action]

        # Update rewards
        success_ratios = [
            threshold / 100 for threshold in round_agent_0["observation"]["thresholds"]
        ]

    return data


if __name__ == "__main__":

    with open(RESULT_DIR, "rb") as file_handler:
        results = pickle.load(file_handler)

    all_dfs = []
    n_games = len(list(results.values())[0])

    progress_bar = tqdm(total=sum([len(elem) for elem in results.values()]))

    for results in results.values():
        for this_round in results:
            all_dfs.append(create_training_data(result=this_round))
            progress_bar.update()

    file_name = (
        f'data/data_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.parquet'
    )
    print(f"Writing results into {file_name}")

    pd.concat(all_dfs).to_parquet(file_name)
