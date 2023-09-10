import argparse
import json
import os
from typing import Dict, Tuple

config_location = os.path.join(
    "network_gym_client", "envs", "nqos_split", "config.json"
)


def get_agent_from_args() -> Tuple[str, bool]:
    parser = argparse.ArgumentParser(description="Agent to Use")
    parser.add_argument(
        "--agent", help="Specify an agent to train with", default="system_default"
    )
    parser.add_argument("--test", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    agent_type: str = args.agent
    testing: bool = args.test
    return (agent_type, testing)


def load_json_file(filename: str) -> Dict:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    full_config_filename = os.path.join(this_file_dir, "..", filename)
    with open(full_config_filename, "r") as json_file:
        json_dict: Dict = json.load(json_file)
    return json_dict


def edit_dict(json_dict: Dict, agent: str, testing: bool = False) -> None:
    json_dict["rl_config"]["agent"] = agent
    training: bool = not testing
    json_dict["rl_config"]["train"] = training
    # FIXME: trying 200k steps to see if PPO learns better (or worse?)
    json_dict["env_config"]["steps_per_episode"] = 400_000 if training else 10_000
    # FIXME: be careful with seeds
    json_dict["env_config"]["random_seed"] = 15 if training else 13


def remove_file(filename: str) -> None:
    if os.path.exists(filename):
        os.remove(filename)


def save_json(config_location: str, json_dict: Dict) -> None:
    with open(config_location, "w") as file:
        json.dump(json_dict, file, indent=4)


def main() -> None:
    agent, testing = get_agent_from_args()
    json_dict = load_json_file(config_location)
    edit_dict(json_dict, agent, testing)
    remove_file(config_location)
    save_json(config_location, json_dict)


if __name__ == "__main__":
    main()
