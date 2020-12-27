import pickle
import random

import numpy as np
import pandas as pd
import sklearn.tree as skt

# Parameters
FUDGE_FACTOR = 0.99
DATA_FILE = "/tmp/data_kaggle.parquet"
TRAIN_FEATS = ["round_num", "n_pulls_self", "n_success_self", "n_pulls_opp"]
TARGET_COL = "payout"


def make_model():
    """Builds a decision tree model based on stored trainingd data"""
    data = pd.read_parquet(DATA_FILE)
    model = skt.DecisionTreeRegressor(min_samples_leaf=40)
    model.fit(data[TRAIN_FEATS], data[TARGET_COL])
    return model


class GreedyStrategy:
    def __init__(self, agent_num, n_machines):
        # Record inputs
        self.agent_num = agent_num
        self.n_machines = n_machines

        # Initialize distributions for all machines
        self.n_pulls_self = np.array([0 for _ in range(n_machines)])
        self.n_success_self = np.array([0.0 for _ in range(n_machines)])
        self.n_pulls_opp = np.array([0 for _ in range(n_machines)])

        # Track other players moves
        self.opp_moves = []

        # Track winnings
        self.last_reward_count = 0

        # Create model to predict expected reward
        self.model = make_model()

        # Predict expected reward
        features = np.zeros((self.n_machines, 4))
        features[:, 0] = len(self.opp_moves)
        features[:, 1] = self.n_pulls_self
        features[:, 2] = self.n_success_self
        features[:, 3] = self.n_pulls_opp
        self.predicts = self.model.predict(features)

    def __call__(self):
        # Otherwise, use best available

        est_return = self.predicts
        max_return = np.max(est_return)

        # Sampling from all rewards that are within 1 -
        result = np.random.choice(np.where(est_return >= FUDGE_FACTOR * max_return)[0])

        return int(result)

    def updateDist(self, curr_total_reward, last_m_indices):
        """Updates estimated distribution of payouts"""
        # Compute last reward
        last_reward = curr_total_reward - self.last_reward_count
        self.last_reward_count = curr_total_reward

        if len(last_m_indices) == 2:
            # Update number of pulls for both machines
            m_index = last_m_indices[self.agent_num]
            opp_index = last_m_indices[(self.agent_num + 1) % 2]
            self.n_pulls_self[m_index] += 1
            self.n_pulls_opp[opp_index] += 1

            # Update number of successes
            self.n_success_self[m_index] += last_reward

            # Update opponent activity
            self.opp_moves.append(opp_index)

            # Update predictions for chosen machines
            self.predicts[[opp_index, m_index]] = self.model.predict(
                [
                    [
                        len(self.opp_moves),
                        self.n_pulls_self[opp_index],
                        self.n_success_self[opp_index],
                        self.n_pulls_opp[opp_index],
                    ],
                    [
                        len(self.opp_moves),
                        self.n_pulls_self[m_index],
                        self.n_success_self[m_index],
                        self.n_pulls_opp[m_index],
                    ],
                ]
            )


def agent(observation, configuration):
    global agent

    if observation.step == 0:
        # Initialize agent
        agent = GreedyStrategy(
            observation["agentIndex"],
            configuration["banditCount"],
        )

    # Update payout ratio distribution with:
    agent.updateDist(observation["reward"], observation["lastActions"])

    return agent()