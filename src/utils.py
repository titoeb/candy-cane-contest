from typing import List, Callable
import multiprocessing as mp
from kaggle_environments import make, evaluate
from typing import List, Dict
import numpy as np


def play_mab(agents: List[str]) -> List[int]:
    """
    Play a round of multi-armed bandit with the specified agents. It returns the trajectory of the game
    """

    # Create environment
    env = make("mab", debug=True)

    # Run environment
    return env.run(agents)


def simulate_mab(agents: List[str], n_rounds: int, n_processes: int) -> List[List[int]]:
    """
    Play multiple rounds of mab and return the single outcomes.
    It returns a list of the steps of all games.
    """

    if n_processes > 1 and n_rounds > 1:
        pool = mp.Pool(n_processes)
        steps_per_game = pool.starmap(
            play_mab,
            [(agents,) for _ in range(n_rounds)],
        )
        pool.close()
    else:
        # No multiprocessing
        steps_per_game = [play_mab(agents) for _ in range(n_rounds)]

    return steps_per_game


def print_sep(sep_length: int = 100) -> str:
    return "-" * 100


def get_agent_final_rewards(agent_idx: int, performances: Dict[str, List]) -> np.array:
    # Extract all rewards
    all_rewards = []
    for match in performances.values():
        for reward in get_rewards_match(agent_idx, match):
            all_rewards.append(reward)
    return np.array(all_rewards)


def get_agent_wins(agent_idx: int, performances: Dict[str, List]) -> np.array:
    # Extract all rewards
    all_wins = []
    for match in performances.values():
        for win in get_wins_match(agent_idx, match):
            all_wins.append(win)
    return np.array(all_wins)


def get_rewards_match(agent_idx: int, match: List) -> int:
    return np.array([session[-1][agent_idx]["reward"] for session in match])


def get_wins_match(agent_idx: int, match: List) -> np.array:
    return np.array(
        [
            session[-1][agent_idx]["reward"] > session[-1][1 - agent_idx]["reward"]
            for session in match
        ]
    )


def get_summary_matches(agent_idx: int, performances: Dict[str, List]) -> Dict:

    summary = {}
    for opponent, match in performances.items():
        try:
            rewards_agent_0 = get_rewards_match(0, match)
            rewards_agent_1 = get_rewards_match(1, match)
            wins_agent_0 = get_wins_match(0, match)
            wins_agent_1 = get_wins_match(1, match)
            agent_0_has_won = wins_agent_0.sum() > wins_agent_1.sum()
        except Exception as e:
            raise ValueError(
                f"An error occured when creating the summary for opponent {opponent}. Error was: {e}"
            )

        summary[opponent] = [
            [rewards_agent_0.mean(), rewards_agent_1.mean()],
            [wins_agent_0.mean(), wins_agent_1.mean()],
            agent_0_has_won,
        ]
    return summary


def create_replace_file(old_file: str, new_file: str, params: Dict):
    # Load old file
    with open(old_file) as file_handler:
        while not "# PARAMS-END #" in file_handler.readline():
            continue
        code = "".join(file_handler.readlines())

    # ADD the params dict in only upper case in the begginning of the file.
    hyper_params = []
    for param_name, value in params.items():
        hyper_params.append(f"{param_name.upper()} = {value}")

    # Write out the file.
    code_output = "\n".join(hyper_params) + "\n" + code

    with open(new_file, "w") as file_handler:
        file_handler.write(code_output)
