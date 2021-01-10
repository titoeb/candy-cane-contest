import numpy as np
import optuna
import datetime
from pathlib import Path
from utils import create_replace_file, get_summary_matches
from evaluate import evaluate_against_opponents
import os

# Params
OPPONENT_POOL = [
    "thompson_sampling",
    "ml_agent_decision_tree",
    "baysian_ucb",
    "vegas_slot_machines",
    "ucb",
    "vegas_slot_machines_2_0_copied",
]
N_EVALS = 200
N_ROUNDS_PER_AGENT = 48
N_PROCESSES = 16

# Params


def optimize(trial):
    # Draw params
    params = {}
    params["RANDOM_TRESHOLD"] = trial.suggest_float(
        "RANDOM_TRESHOLD", 0.0, 5.0, log=False
    )
    params["C"] = trial.suggest_float("C", 0.0, 1.0, log=False)
    # Set them in file
    create_replace_file(
        old_file="agents/vegas_slot_machines_2_0.py",
        new_file="agents/tmp.py",
        params=params,
    )

    # Run candidate against pool
    _, results = evaluate_against_opponents(
        candidate_path=Path("agents/tmp.py"),
        opponents_paths=[Path(f"agents/{opponent}.py") for opponent in OPPONENT_POOL],
        n_rounds_per_agent=N_ROUNDS_PER_AGENT,
        n_processes=N_PROCESSES,
    )
    summary = get_summary_matches(0, results)

    # Report win-ratio as target variable to maximize
    return np.array([elem[1][0] for elem in summary.values()]).mean()


study = optuna.create_study(
    storage=f'sqlite:///optuna/run_{datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")}.db',
    direction="maximize",
    study_name="optimize_ucb_generic",
)

study.optimize(optimize, n_trials=N_EVALS)

# Clean up the written tmp.file
os.remove("agents/tmp.py")