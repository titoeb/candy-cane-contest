import json
import typer
from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import Tuple, Dict
import datetime
import pickle
import numpy as np


def load_json(path: pathlib.Path) -> Dict:
    with open(path) as file_handler:
        data = file_handler.read()

    return json.loads(data)


def extract_replay(play: Dict) -> Tuple[str, Dict]:
    return play["info"]["TeamNames"][0], play["steps"]


def extract_top_replays(replays_folder: str, data_folder: str):
    replays_path = Path(replays_folder)
    all_replays = [
        this_file
        for this_file in replays_path.iterdir()
        if this_file.is_file() and this_file.name.endswith(".json")
    ]

    data = dict()

    for replay in tqdm(all_replays):
        agent_name, game = extract_replay(load_json(replay))
        if not data.get(agent_name):
            data[agent_name] = [game]
        else:
            data[agent_name] += [game]

    # Store the extracted runs
    file_name = f'{data_folder}/extracted_top_plays_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pickle'

    with open(file_name, "wb") as file_handler:
        pickle.dump(data, file_handler)

    print(f"Store results to {file_name}")


def get_final_rewards_sum(replay: Dict):
    if replay["steps"][-1][0]["reward"] and replay["steps"][-1][1]["reward"]:
        return replay["steps"][-1][0]["reward"] + replay["steps"][-1][1]["reward"]
    else:
        return 0


def main(replays_folder: str, data_folder: str, topn: int):

    # Identify the games with the highest rewards.
    replays_path = Path(replays_folder)
    all_replays = [
        this_file
        for this_file in replays_path.iterdir()
        if this_file.is_file() and this_file.name.endswith(".json")
    ]

    all_rewards = np.array(
        [get_final_rewards_sum(load_json(replay)) for replay in tqdm(all_replays)]
    )

    n_th_reward = np.sort(all_rewards)[::-1][topn]

    rel_games = np.where(all_rewards >= n_th_reward)[0]

    data = dict()

    for replay_idx in tqdm(rel_games):
        agent_name, game = extract_replay(load_json(all_replays[replay_idx]))
        if not data.get(agent_name):
            data[agent_name] = [game]
        else:
            data[agent_name] += [game]

    # Store the extracted runs
    file_name = f'{data_folder}/extracted_top_plays_{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.pickle'

    with open(file_name, "wb") as file_handler:
        pickle.dump(data, file_handler)

    print(f"Store results to {file_name}")


if __name__ == "__main__":
    # typer.run(main)
    main("top-replays", "data", topn=1000)
