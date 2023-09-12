import numpy as np
import os
import pandas as pd
from copy import deepcopy
from typing import Dict, List

COUNTER = 0
FILENAME = "shape_log.txt"

NAME_MAP = {
    "max_rate": ["LTE", "Wi-Fi"],
    "tx_rate": ["All"],
    "rate": ["LTE", "Wi-Fi"],
    "owd": ["LTE", "Wi-Fi"],
    "max_owd": ["LTE", "Wi-Fi"],
    "cell_id": ["Wi-Fi"],
    "traffic_ratio": ["LTE", "Wi-Fi"],
    "x_loc": ["All"],
    "y_loc": ["All"],
}

# name -> channel id -> values
PREVIOUS_ENTRIES: Dict[str, Dict[str, List[float]]] = {}

PREVIOUS_SPLIT_RATIOS: List[float] = []


def get_previous_action(df: pd.DataFrame) -> List[float]:
    global PREVIOUS_SPLIT_RATIOS
    split_ratio_df = df[df["name"] == "split_ratio"]
    split_ratio_lte_df = split_ratio_df[split_ratio_df["cid"] == "Wi-Fi"]
    previous_split_ratios = deepcopy(PREVIOUS_SPLIT_RATIOS)
    if not split_ratio_lte_df.empty:
        new_users = split_ratio_lte_df["user"].values[0]
        new_values = split_ratio_lte_df["value"].values[0]
        for user, value in zip(new_users, new_values):
            previous_split_ratios[user] = value / 32.0
    PREVIOUS_SPLIT_RATIOS = deepcopy(previous_split_ratios)
    return previous_split_ratios


def turn_df_into_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    # with pd.option_context(
    #     "display.max_rows", None, "display.max_columns", None
    # ):
    #     print(df)
    df_list = [df[df["name"] == name] for name in NAME_MAP]
    return df_list


def print_full_observation(obs_array_list: List[np.ndarray]) -> None:
    print(np.vstack(obs_array_list))


def initialize_previous_entries(num_users: int) -> None:
    global PREVIOUS_SPLIT_RATIOS
    for name, channels in NAME_MAP.items():
        PREVIOUS_ENTRIES[name] = {}
        name_dict = PREVIOUS_ENTRIES[name]
        for channel_id in channels:
            values = [0.0] * num_users
            name_dict[channel_id] = values
    PREVIOUS_SPLIT_RATIOS = [1.0] * num_users


def get_full_observation(df_list: List[pd.DataFrame]) -> List[np.ndarray]:
    num_users = len(df_list[0]["value"].values[0])
    if len(PREVIOUS_ENTRIES) == 0:
        initialize_previous_entries(num_users)
    full_observation: List[np.ndarray] = []
    for df, channels in zip(df_list, NAME_MAP.values()):
        names: List[str] = df["name"].values
        name = names[0]
        for channel in channels:
            # start with the previous row by default and overwrite with new information
            previous_values = PREVIOUS_ENTRIES[name][channel]
            previous_row = np.array(previous_values)
            row = df[df["cid"] == channel]
            if not row.empty:
                users = row["user"].values[0]
                values = row["value"].values[0]
                for user, value in zip(users, values):
                    previous_row[user] = value
            # NOTE: fix for access point ID being too small relative to other values
            if name == "cell_id":
                previous_row *= 100
            full_observation.append(previous_row)
    repopulate_previous_entries(df_list)
    return full_observation


def repopulate_previous_entries(df_list: List[pd.DataFrame]) -> None:
    for df in df_list:
        for _, row in df.iterrows():
            name: str = row["name"]
            channel: str = row["cid"]
            if name in PREVIOUS_ENTRIES and channel in PREVIOUS_ENTRIES[name]:
                previous_values = PREVIOUS_ENTRIES[name][channel]
                users = row["user"]
                values = row["value"]
                for user, value in zip(users, values):
                    # NOTE: fix for rate or traffic ratio being absent
                    if "rate" in name or "traffic_ratio" in name:
                        previous_values[user] = 0
                    else:
                        previous_values[user] = value


def log_full_observation(df: pd.DataFrame) -> None:
    global COUNTER
    log_line(f"COUNTER: {COUNTER}")
    names = df["name"]
    frequencies: Dict[str, int] = {}
    for name in names:
        frequencies[name] = frequencies.get(name, 0) + 1
    for key, value in frequencies.items():
        log_line(f"{key},{value}")
    COUNTER += 1


def log_line(line: str) -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(this_dir, "..", FILENAME)
    with open(log_path, "a") as log:
        log.write(line + "\n")
