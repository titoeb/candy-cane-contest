ML_MODEL_BASE_FILE = "/usr/src/ml-models-base/greedy_base.py"
MODEL_FILE = "/usr/src/models/decision_tree.txt"
AGENT_FILE_PATH = "/usr/src/agents/ml_agent_decision_tree.py"

with open(ML_MODEL_BASE_FILE) as file_handler:
    ml_model_base = file_handler.read()


with open(MODEL_FILE) as file_handler:
    ml_model = file_handler.read()


final_agent = ml_model + "\n" + ml_model_base

with open(AGENT_FILE_PATH, "w") as file_handler:
    file_handler.write(final_agent)
