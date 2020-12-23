"""Agent that samples the estimated payout ratio distribution.

The a posteriori distribution of potential payout ratios for each machine
is tracked and updated based on the results of each pull. The a priori
distribution for each machine is a uniform distribution from 0.0 to 1.0.

When selecting a machine to pull, each distribution is sample a configurable
number of times.  The machine with the sample(s) that generate the highest
expected reward is selected for the next pull.

"""
import random

import numpy as np

# Parameters
NUM_SAMPLES = 5
PRIOR_DISTRIBUTION = np.array([0.01] * 100)


class MonteCarloStragegy:
    """Implements strategy to maximize expected value

    - Tracks estimated likelihood of payout ratio for each machine
    - Tracks number of pulls on each machine
    - Chooses machine based on maximum reward from a limited Monte-Carlo
      simulation based on the estimated distribution of payout ratios


    """

    def __init__(self, name, agent_num, initial_dist, ev_rounds, n_machines):
        """Initialize with simple distribution of payout probabilities

        Args:
           name (str):   Name for the agent
           agent_num (int):   Assigned player number
           initial_dist (np.array, (100,)):   a priori payout distribution for
               each machine.
           ev_rounds (int):   number of samples to average for monte-carlo
               expected value calculation
           n_machines (int):   number of machines in the game

        """
        # Record inputs
        self.name = name
        self.agent_num = agent_num
        self.initial_dist = initial_dist
        self.ev_rounds = ev_rounds  # Num rounds to base MC choice on
        self.n_machines = n_machines

        # Initialize discrete set of payout ratios
        self.p_ratios = np.linspace(0, 0.99, 100)

        # Initialize distributions for all machines
        self.n_pulls = [0 for _ in range(n_machines)]
        self.dist = [initial_dist for m_index in range(n_machines)]
        self.cum_dist = [self.updateCumDist(m_index) for m_index in range(n_machines)]

        # Track winnings!
        self.last_reward_count = 0

    def __call__(self):
        """Choose machine based on maximum Monte-Carlo return

        Returns:
           <result> (int):  index of machine to pull

        """
        # Select machine with highest return on limited Monte Carlo
        est_return = np.array(
            [self.estimatedReturn(m_index) for m_index in range(self.n_machines)]
        )
        return int(np.argmax(est_return))

    def samplePayoutRatio(self, m_index):
        """Pull a weighted sample from the distribution"""
        x = random.random()
        return self.p_ratios[np.where(x <= self.cum_dist[m_index])[0][0]]

    def estimatedReturn(self, m_index):
        """Expected return from a Monte-Carlo sample of payout ratios"""
        n_pulls = self.n_pulls[m_index]
        est_p = sum([self.samplePayoutRatio(m_index) for ii in range(self.ev_rounds)])
        return est_p / self.ev_rounds * 0.97 ** n_pulls

    def updateDist(self, curr_total_reward, last_m_indices):
        """Updates estimated distribution of payouts"""
        # Compute last reward
        last_reward = curr_total_reward - self.last_reward_count
        self.last_reward_count = curr_total_reward

        if len(last_m_indices) == 2:
            # Update number of pulls for both machines
            self.n_pulls[last_m_indices[0]] += 1
            self.n_pulls[last_m_indices[1]] += 1

            # Update estimated probabilities for this agent's pull
            m_index = last_m_indices[self.agent_num]
            n_pulls = self.n_pulls[m_index]
            if last_reward == 1:
                curr_prob = self.p_ratios * 0.97 ** n_pulls
            else:
                curr_prob = 1 - self.p_ratios * 0.97 ** n_pulls

            self.dist[m_index] = curr_prob * self.dist[m_index]
            self.dist[m_index] = self.dist[m_index] / self.dist[m_index].sum()
            self.cum_dist[m_index] = self.updateCumDist(m_index)

    def updateCumDist(self, m_index):
        """Updates cumulative payout ratio distribution"""
        return np.cumsum(self.dist[m_index])


# DEFINE AGENT ----------------------------------------------------------------


def agent(observation, configuration):
    global curr_agent

    if observation.step == 0:
        # Initialize agent
        curr_agent = MonteCarloStragegy(
            "Mr. Agent %i" % observation["agentIndex"],
            observation["agentIndex"],
            PRIOR_DISTRIBUTION,
            NUM_SAMPLES,
            configuration["banditCount"],
        )

    # Update payout ratio distribution with:
    # - which machines were pulled by both players
    # - result from previous pull
    curr_agent.updateDist(observation["reward"], observation["lastActions"])

    return curr_agent()
