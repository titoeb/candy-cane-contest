import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import base64
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from functools import partial


def encode_base64_string(model: Any) -> str:
    return base64.b64encode(pickle.dumps(model))


def optimize_decision_tree(trial, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    max_depth = trial.suggest_categorical("use_max_depth", [None, 1])
    if max_depth:
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100, log=True)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 100, log=True)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

    # Train model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    model.fit(X_train, y_train)

    # Test model on test data
    test_preds = model.predict(X_test)

    return np.square(test_preds - y_test).mean()


DATA_FILE = "data/data_initial.parquet"

# DATA_FILE = "data/data_initial.parquet"
MODEL_FILE = "/usr/src/models/decision_tree.txt"
# TRAIN_FEATS = ["round", "n_pulls_self", "n_success_self", "n_pulls_opponent"]
# TARGET_COL = "success_probs"
N_TRIALS = 50
RANDOM_STATE = 1993
PERCENTAGE_TEST = 0.7


TRAIN_FEATS = ["round_num", "n_pulls_self", "n_success_self", "n_pulls_opp"]
TARGET_COL = "payout"

# Load data
data = pd.read_parquet(DATA_FILE)

X = data[TRAIN_FEATS]
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=RANDOM_STATE, test_size=PERCENTAGE_TEST
)


print(f"{X_train.shape[0]} out of {data.shape[0]} observations were used for training")

# Determine optimal parameters with optuna
study = optuna.create_study()
study.optimize(partial(optimize_decision_tree, X=X, y=y), n_trials=N_TRIALS)
optimal_params = study.best_trial.params
_ = optimal_params.pop("use_max_depth")

# Train model
model = DecisionTreeRegressor(**optimal_params)
model.fit(X_train, y_train)

# Test model on test data
test_preds = model.predict(X_test)

mse = np.square(test_preds - y_test).mean()

print(f"MSE: {mse}")


# Store model as base64 string that will be prependet to agents
# file
model_string = encode_base64_string(model)

final_string = f"model = {model_string}"

with open(MODEL_FILE, "w") as file_handler:
    file_handler.write(final_string)
