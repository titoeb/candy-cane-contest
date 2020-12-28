import pandas as pd
import numpy as np
import pickle
from typing import Dict, List


RESULT_DIR = "/usr/src/data/results_2020-12-28--08-08-32.pickle"


def create_training_data(
    result: List[Dict],
) -> pd.DataFrame:

    # Get number of bins and opponents from the data
    n_bins = result[0].n_machines

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
    data.iloc[:1000]["success_probs"] = [
        threshold / 100 for threshold in result[0]["thresholds"]
    ]

    agent_0_state = {
        "n_pulls_self": np.full(n_bins, 0.0),
        "n_success_self": np.full(n_bins, 0.0),
        "n_pulls_opponent": np.full(n_bins, 0.0),
    }

    agent_1_state = {
        "n_pulls_self": np.full(n_bins, 0.0),
        "n_success_self": np.full(n_bins, 0.0),
        "n_pulls_opponent": np.full(n_bins, 0.0),
    }

    success_ratios = np.full(n_bins, 0.0)

    for round in result:
        if round["step"] > 0:
            # Update agents
            pass

            # Insert one datapoint for each agent

        # Update rewards
        success_ratios = [threshold / 100 for threshold in round["thresholds"]]


if __name__ == "__main__":

    with open(RESULT_DIR, "rb") as file_handler:
        results = pickle.load(file_handler)

    df = pd.concat(
        [create_training_data(result=results) for results in results.values()]
    )
