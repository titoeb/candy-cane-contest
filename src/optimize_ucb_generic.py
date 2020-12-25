from evaluate import main as evaluate_main
import numpy as np
import re
import optuna
import datetime

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
# Params

def optimize_ucb_generic(trial):
    # Draw params
    
    # Set them in file 

    # Run candidate against pool

    # Report 1 - win-ratio as target variable
    return 1.0

study = optuna.create_study(storage=f"sqlite:///optuna/run_{strftime("%d-%m-%Y--%H-%M-%S")}.db")
study.optimize(optimize_ucb_generic n_trials=N_EVALS)