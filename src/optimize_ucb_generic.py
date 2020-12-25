from evaluate import main as evaluate_main
import numpy as np
import re
import optuna
import datetime
from utils import create_and_replace_file, get_summary_matches
from evaluate import evaluate_against_opponents

# Params
OPPONENT_POOL = [
    "thompson",
    "baysian_ucb",
    "vegas_slot_machine",
    "monte-carlo",
    "ucb_decaying",
    "ucb",
]
N_EVALS = 2
N_ROUNDS_PER_AGENT = 1
N_PROCESSES = 1

# Params


def optimize_ucb_generic(trial):
    # Draw params
    params = {}
    params["exploration"] = trial.suggest_float("exploration", 1e-8, 100.0, log=True)
    params["sampling"] = trial.suggest_categorical("sampling", [True, False])
    params["decaying"] = trial.suggest_float("decaying", 0.5, 2.5)
    if trial.suggest_categorical("use_opponent_actions", [True, False]):
        params["damping_factor"] = trial.suggest_float("damping_factor", 1e-8, 1000.0)
        params["buffer_length"] = trial.suggest_int("buffer_length", 0, 2000)
        params["min_step_opponent"] = trial.suggest_int("min_step_opponent", 0, 2000)
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
        candidate_path="agents/ucb_generic.py",
        opponent_paths=OPPONENT_POOL,
        n_rounds_per_agent=N_ROUNDS_PER_AGENT,
        n_processes=N_PROCESSES,
    )
    summary = get_summary_matches(0, results)

    # Report 1 - win-ratio as target variable
    return summary[1][0]


study = optuna.create_study(
    storage=f'sqlite:///optuna/run_{datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")}.db',
    direction="maximize",
)
study.optimize(optimize_ucb_generic, n_trials=N_EVALS)

# Clean up the written tmp.file
