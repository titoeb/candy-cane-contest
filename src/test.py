from utils import play_mab, simulate_mab
from agent_random import agent as agent_random
from agent_vegas_slot_machines import agent as agent_slot_machine
from agent_kaggle_epsilon_greedy_decay import multi_armed_probabilities

if __name__ == "__main__":
    # greedy_agent = GreedyAgent(n_bins=100)
    # agent = EpsilonDecayingGreedyAgent(
    # n_bins=100, epsilon_start=1.0, ratio_decay=0.9997, min_epsilon=0.4
    # )

    AGENTS = [multi_armed_probabilities, agent_slot_machine]

    # print(play_mab(agents=[cycle_agent, greedy_agent.play]))
    res = simulate_mab(agents=AGENTS, n_rounds=1, n_processes=1)
    print(res)
