import argparse
import json
import os
from config_lock.client_utils import request_lock

config_location = os.path.join(
    "network_gym_client", "envs", "nqos_split", "config.json"
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="For changing nqos_split/config.json")
    parser.add_argument(
        "--agent", help="rl_config['agent'] in config.json", required=False, type=str
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="set rl_config['train'] = true in config.json",
        required=False,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="set rl_config['train'] = false in config.json",
        required=False,
    )
    parser.add_argument(
        "--steps",
        help="env_config['steps_per_episode'] in config.json",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="env_config['random_seed'] in config.json",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--store_offline",
        action="store_true",
        help="set rl_config['store_offline'] = true in config.json",
        required=False,
    )
    args = parser.parse_args()
    return args


def load_json_file(filename: str) -> dict:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    full_config_filename = os.path.join(this_file_dir, "..", filename)
    with open(full_config_filename, "r") as json_file:
        json_dict: dict = json.load(json_file)
    return json_dict


def edit_dict(json_dict: dict, args: argparse.Namespace) -> None:
    agent: str | None = args.agent
    training: bool = args.train
    testing: bool = args.test
    store_offline: bool = args.store_offline
    steps: int | None = args.steps
    seed: int | None = args.seed

    if agent != None:
        json_dict["rl_config"]["agent"] = agent

    if training and testing:
        raise Exception(
            "--train and --test can't both be supplied to config_changer.py"
        )
    elif training:
        json_dict["rl_config"]["train"] = True
    elif testing:
        json_dict["rl_config"]["train"] = False

    json_dict["rl_config"]["store_offline"] = store_offline

    if steps != None:
        json_dict["env_config"]["steps_per_episode"] = steps

    if seed != None:
        json_dict["env_config"]["random_seed"] = seed


def remove_file(filename: str) -> None:
    if os.path.exists(filename):
        os.remove(filename)


def save_json(config_location: str, json_dict: dict) -> None:
    with open(config_location, "w") as file:
        json.dump(json_dict, file, indent=4)


def main() -> None:
    args = get_args()
    seed: int | None = args.seed
    if seed != None:
        request_lock(seed)
    json_dict = load_json_file(config_location)
    edit_dict(json_dict, args)
    remove_file(config_location)
    save_json(config_location, json_dict)


if __name__ == "__main__":
    main()
