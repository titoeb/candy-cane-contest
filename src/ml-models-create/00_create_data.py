import pandas as pd
import numpy as np
import pickle
from typing import Dict, List
import datetime


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
            # Action agent 0 from perspective of agent 0
            row = n_bins + (current_step - 1) * 4
            data.loc[
                row,
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_0_state["n_pulls_self"][agent_0_action],
                agent_0_state["n_success_self"][agent_0_action],
                agent_0_state["n_pulls_opponent"][agent_0_action],
                success_ratios[agent_0_action],
            ]

            # Action agent 0 from perspective of agent 1
            data.loc[
                row + 1,
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_1_state["n_pulls_self"][agent_0_action],
                agent_1_state["n_success_self"][agent_0_action],
                agent_1_state["n_pulls_opponent"][agent_0_action],
                success_ratios[agent_0_action],
            ]

            # Action agent 1 from agent 1 perspective
            data.loc[
                row + 2,
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_1_state["n_pulls_self"][agent_1_action],
                agent_1_state["n_success_self"][agent_1_action],
                agent_1_state["n_pulls_opponent"][agent_1_action],
                success_ratios[agent_1_action],
            ]

            # Action agent 1 from agent 0 perspective
            data.loc[
                row + 3,
                [
                    "n_pulls_self",
                    "n_success_self",
                    "n_pulls_opponent",
                    "success_probs",
                ],
            ] = [
                agent_0_state["n_pulls_self"][agent_1_action],
                agent_0_state["n_success_self"][agent_1_action],
                agent_0_state["n_pulls_opponent"][agent_1_action],
                success_ratios[agent_1_action],
            ]

        # Update rewards
        success_ratios = [
            threshold / 100 for threshold in round_agent_0["observation"]["thresholds"]
        ]

    return data


def log_training(result, n_machines):
    """Records training data from each machine, each agent, each round

    Generates a training dataset to support prediction of the current
    payout ratio for a given machine.

    Args:
       result ([[dict]]) - output from all rounds provided as output of
                           env.run([agent1, agent2])
       n_machines (int) - number of machines

    Returns:
       training_data (pd.DataFrame) - training data, including:
           "round_num"      : round number
           "machine_id"     : machine data applies to
           "agent_id"       : player data applies to (0 or 1)
           "n_pulls_self"   : number of pulls on this machine so far by agent_id
           "n_success_self" : number of rewards from this machine by agent_id
           "n_pulls_opp"    : number of pulls on this machine by the other player
           "payout"         : actual payout ratio for this machine

    """
    # Initialize machine and agent states
    machine_state = [
        {
            "n_pulls_0": 0,
            "n_success_0": 0,
            "n_pulls_1": 0,
            "n_success_1": 0,
            "payout": None,
        }
        for ii in range(n_machines)
    ]
    agent_state = {"reward_0": 0, "reward_1": 0, "last_reward_0": 0, "last_reward_1": 0}

    # Initialize training dataframe
    # - In the first round, store records for all n_machines
    # - In subsequent rounds, just store the two machines that updated
    training_data = pd.DataFrame(
        index=range(n_machines + 4 * (len(result) - 1)),
        columns=[
            "round_num",
            "machine_id",
            "agent_id",
            "n_pulls_self",
            "n_success_self",
            "n_pulls_opp",
            "payout",
        ],
    )

    # Log training data from each round
    for round_num, res in enumerate(result):
        # Get current threshold values
        thresholds = res[0]["observation"]["thresholds"]

        # Update agent state
        for agent_ii in range(2):
            agent_state["last_reward_%i" % agent_ii] = (
                res[agent_ii]["reward"] - agent_state["reward_%i" % agent_ii]
            )
            agent_state["reward_%i" % agent_ii] = res[agent_ii]["reward"]

        # Update most recent machine state
        if res[0]["observation"]["lastActions"]:
            for agent_ii, r_obs in enumerate(res):
                action = r_obs["action"]
                machine_state[action]["n_pulls_%i" % agent_ii] += 1
                machine_state[action]["n_success_%i" % agent_ii] += agent_state[
                    "last_reward_%i" % agent_ii
                ]
                machine_state[action]["payout"] = thresholds[action]
        else:
            # Initialize machine states
            for mach_ii in range(n_machines):
                machine_state[mach_ii]["payout"] = thresholds[mach_ii]

        # Record training records
        # -- Each record includes:
        #       round_num, n_pulls_self, n_success_self, n_pulls_opp
        if res[0]["observation"]["lastActions"]:
            # Add results for most recent moves
            for agent_ii, r_obs in enumerate(res):
                action = r_obs["action"]

                # Add row for agent who acted
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii
                training_data.at[row_ii, "round_num"] = round_num
                training_data.at[row_ii, "machine_id"] = action
                training_data.at[row_ii, "agent_id"] = agent_ii
                training_data.at[row_ii, "n_pulls_self"] = machine_state[action][
                    "n_pulls_%i" % agent_ii
                ]
                training_data.at[row_ii, "n_success_self"] = machine_state[action][
                    "n_success_%i" % agent_ii
                ]
                training_data.at[row_ii, "n_pulls_opp"] = machine_state[action][
                    "n_pulls_%i" % ((agent_ii + 1) % 2)
                ]
                training_data.at[row_ii, "payout"] = (
                    machine_state[action]["payout"] / 100
                )

                # Add row for other agent
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii + 1
                other_agent = (agent_ii + 1) % 2
                training_data.at[row_ii, "round_num"] = round_num
                training_data.at[row_ii, "machine_id"] = action
                training_data.at[row_ii, "agent_id"] = other_agent
                training_data.at[row_ii, "n_pulls_self"] = machine_state[action][
                    "n_pulls_%i" % other_agent
                ]
                training_data.at[row_ii, "n_success_self"] = machine_state[action][
                    "n_success_%i" % other_agent
                ]
                training_data.at[row_ii, "n_pulls_opp"] = machine_state[action][
                    "n_pulls_%i" % agent_ii
                ]
                training_data.at[row_ii, "payout"] = (
                    machine_state[action]["payout"] / 100
                )

        else:
            # Add initial data for all machines
            for action in range(n_machines):
                row_ii = action
                training_data.at[row_ii, "round_num"] = round_num
                training_data.at[row_ii, "machine_id"] = action
                training_data.at[row_ii, "agent_id"] = -1
                training_data.at[row_ii, "n_pulls_self"] = 0
                training_data.at[row_ii, "n_success_self"] = 0
                training_data.at[row_ii, "n_pulls_opp"] = 0
                training_data.at[row_ii, "payout"] = (
                    machine_state[action]["payout"] / 100
                )

    return training_data


if __name__ == "__main__":

    with open(RESULT_DIR, "rb") as file_handler:
        results = pickle.load(file_handler)

    # df = pd.concat(
    #     [create_training_data(result=results[0]) for results in results.values()]
    # )

    df = pd.concat(
        [
            pd.concat(
                [
                    log_training(result=this_round, n_machines=100)
                    for this_round in results
                ]
            )
            for results in results.values()
        ]
    )

    file_name = (
        f'data/data_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.parquet'
    )
    df.to_parquet(file_name)
