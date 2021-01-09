import pandas as pd
import pickle
import base64
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_opt


def encode_base64_string(model: Any) -> str:
    return base64.b64encode(pickle.dumps(model))


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

dtrain = lgb_opt.Dataset(X_train, label=y_train)
dval = lgb_opt.Dataset(X_test, label=y_test)

params = {
    "objective": "regression",
    "metric": "mse",
    "verbosity": -1,
}

model = lgb_opt.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    verbose_eval=5,
    early_stopping_rounds=100,
)


best_params = model.params
# Test model on test data
test_preds = model.predict(X_test)

mse = np.square(test_preds - y_test).mean()

print(f"MSE: {mse}")

# Train final model on all data with correct number of estimators:
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(
    X,
    y,
)


# Store model as base64 string that will be prependet to agents
# file
model_string = encode_base64_string(final_model)

final_string = f"model = {model_string}"

with open(MODEL_FILE, "w") as file_handler:
    file_handler.write(final_string)
