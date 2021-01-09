from utils import play_mab, simulate_mab
from agents.random import agent as random_agent

# from agents.tmp import agent as agent_tmp
# from agents.ml_agent_decision_tree import agent as agent_decision_tree
from agents.vegas_slot_machines_2_0 import agent

if __name__ == "__main__":
    # greedy_agent = GreedyAgent(n_bins=100)
    # agent = EpsilonDecayingGreedyAgent(
    # n_bins=100, epsilon_start=1.0, ratio_decay=0.9997, min_epsilon=0.4
    # )

    AGENTS = [random_agent, agent]

    # print(play_mab(agents=[cycle_agent, greedy_agent.play]))
    res = simulate_mab(agents=AGENTS, n_rounds=1, n_processes=1)
    print(res)
