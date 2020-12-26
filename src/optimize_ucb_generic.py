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
    "baysian_ucb",
    "vegas_slot_machines",
    "monte_carlo",
    "ucb_decaying",
    "ucb",
]
N_EVALS = 300
N_ROUNDS_PER_AGENT = 48
N_PROCESSES = 16

# Params


def optimize_ucb_generic(trial):
    # Draw params
    params = {}
    params["exploration"] = trial.suggest_float("exploration", 1e-8, 100.0, log=True)
    params["sampling"] = trial.suggest_categorical("sampling", [True, False])
    params["decaying"] = trial.suggest_float("decaying", 0.8, 1.2)

    if trial.suggest_categorical("use_opponent_actions", [True, False]) is True:
        params["damping_factor"] = trial.suggest_float("damping_factor", 1e-8, 1000.0)
        params["buffer_length"] = trial.suggest_int("buffer_length", 1, 2000)
        params["min_step_opponent"] = trial.suggest_int("min_step_opponent", 1, 2000)
    else:
        params["damping_factor"] = None
        params["buffer_length"] = 0
        params["min_step_opponent"] = 0

    # Set them in file
    create_replace_file(
        old_file="agents/ucb_generic.py", new_file="agents/tmp.py", params=params
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

study.optimize(optimize_ucb_generic, n_trials=N_EVALS)

# Clean up the written tmp.file
os.remove("agents/tmp.py")