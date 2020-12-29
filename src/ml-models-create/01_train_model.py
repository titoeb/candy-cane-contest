import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import base64
from typing import Any


def encode_base64_string(model: Any) -> str:
    return base64.b64encode(pickle.dumps(model))


# DATA_FILE = "data/data_2020-12-29--08-31-32.parquet"
DATA_FILE = "data/data_initial.parquet"
MODEL_FILE = "/usr/src/models/decision_tree.txt"
# TRAIN_FEATS = ["round", "n_pulls_self", "n_success_self", "n_pulls_opponent"]
# TARGET_COL = "success_probs"

TRAIN_FEATS = ["round_num", "n_pulls_self", "n_success_self", "n_pulls_opp"]
# TARGET_COL = "payout"

TARGET_COL = "sucess_probs"


# Load data
data = pd.read_parquet(DATA_FILE)

# Train model
model = DecisionTreeRegressor(min_samples_leaf=40)
model.fit(data[TRAIN_FEATS], data[TARGET_COL])

# Store model as base64 string that will be prependet to agents
# file
model_string = encode_base64_string(model)

final_string = f"model = {model_string}"

with open(MODEL_FILE, "w") as file_handler:
    file_handler.write(final_string)
