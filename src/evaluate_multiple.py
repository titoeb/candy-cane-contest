from evaluate import main as evaluate_main
import numpy as np
import re


static_agent = "agent_ucb_new.py"
agent_to_search = "agent_vegas_slot_machines.py"


with open(
    agent_to_search,
) as file_handler:
    agent_code = file_handler.read()

for value_to_replace in np.linspace(1.6, 3, num=30):
    code = agent_code.replace("c=0.25", f"c={value_to_replace}")
    with open("tmp.py", "w") as file_handler:
        file_handler.write(code)

    print(f"With hyper parameter value {value_to_replace}:")
    evaluate_main("tmp.py", "agent_ucb_new.py", 160, 16)
    print("\n\n")
