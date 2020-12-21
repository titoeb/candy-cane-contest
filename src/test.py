from utils import play_mab, simulate_mab
from agents import (
    GreedyAgent,
    cycle_agent,
    EpsilonGreedyAgent,
    EpsilonDecayingGreedyAgent,
    ThomsonSampler,
    UpperConfidenceBound,
)

if __name__ == "__main__":
    # greedy_agent = GreedyAgent(n_bins=100)
    # agent = EpsilonDecayingGreedyAgent(
    # n_bins=100, epsilon_start=1.0, ratio_decay=0.9997, min_epsilon=0.4
    # )

    AGENTS = ["/usr/src/submission_thomson.py", "/usr/src/submission_ucb.py"]

    # print(play_mab(agents=[cycle_agent, greedy_agent.play]))
    res = simulate_mab(agents=AGENTS, n_rounds=1, n_processes=1)
    print(res)
