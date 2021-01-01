import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import base64
from typing import Any
from sklearn.model_selection import train_test_split

def encode_base64_string(model: Any) -> str:
    return base64.b64encode(pickle.dumps(model))


DATA_FILE = "data/data_2020-12-29--14-30-28.parquet"
# DATA_FILE = "data/data_initial.parquet"
MODEL_FILE = "/usr/src/models/decision_tree.txt"
TRAIN_FEATS = ["round", "n_pulls_self", "n_success_self", "n_pulls_opponent"]
TARGET_COL = "success_probs"
RANDOM_STATE = 1993
PERCENTAGE_TEST = 0.01


# TRAIN_FEATS = ["round_num", "n_pulls_self", "n_success_self", "n_pulls_opp"]
# TARGET_COL = "payout"

# TARGET_COL = "sucess_probs"


# Load data
data = pd.read_parquet(DATA_FILE)

X = data[TRAIN_FEATS]
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size = PERCENTAGE_TEST)
print(f"{X_train.shape[0]} out of {data.shape[0]} observations were used for training")


# Train model
model = DecisionTreeRegressor(min_samples_leaf=40)
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
