import json
import typer
from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import Tuple, Dict
import datetime
import pickle


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


if __name__ == "__main__":
    # typer.run(extract_top_replays)
    extract_top_replays("top-replays", "data")
