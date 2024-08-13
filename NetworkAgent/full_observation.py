import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict, List

COUNTER = 0
FILENAME = "shape_log.txt"

NAME_MAP = {
    "dl::max_rate": ["lte", "wifi"],
    "dl::tx_rate": ["gma"],
    "lte::dl::rate": ["gma"],
    "wifi::dl::rate": ["gma"],
    "lte::dl::owd": ["gma"],
    "wifi::dl::owd": ["gma"],
    "lte::dl::max_owd": ["gma"],
    "wifi::dl::max_owd": ["gma"],
    "cell_id": ["wifi"],
    "lte::dl::traffic_ratio": ["gma"],
    "wifi::dl::traffic_ratio": ["gma"],
    "x_loc": ["gma"],
    "y_loc": ["gma"],
}

# name -> channel id -> values
PREVIOUS_ENTRIES: Dict[str, Dict[str, List[float]]] = {}

PREVIOUS_SPLIT_RATIOS: List[float] = []


def get_previous_action(df: pd.DataFrame) -> List[float]:
    global PREVIOUS_SPLIT_RATIOS
    split_ratio_wifi_df = df[df["name"] == "wifi::dl::split_ratio"]
    previous_split_ratios = deepcopy(PREVIOUS_SPLIT_RATIOS)
    if not split_ratio_wifi_df.empty:
        new_users = split_ratio_wifi_df["id"].values[0]
        new_values = split_ratio_wifi_df["value"].values[0]
        for user, value in zip(new_users, new_values):
            previous_split_ratios[user - 1] = value / 100.0
    PREVIOUS_SPLIT_RATIOS = deepcopy(previous_split_ratios)
    return previous_split_ratios


def turn_df_into_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
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
    # FIXME HIGH: change this back to 1.0 for accurate values system_default collection
    PREVIOUS_SPLIT_RATIOS = [0.5] * num_users


def get_full_observation(df_list: list[pd.DataFrame], num_users: int) -> list[np.ndarray]:
    if len(PREVIOUS_ENTRIES) == 0:
        initialize_previous_entries(num_users)
    full_observation: List[np.ndarray] = []
    for df, channels in zip(df_list, NAME_MAP.values()):
        names: List[str] = df["name"].values
        assert len(names) == 1
        name = names[0]
        for channel in channels:
            # start with the previous row by default and overwrite with new information
            previous_values = PREVIOUS_ENTRIES[name][channel]
            previous_row = np.array(previous_values)
            row = df[df["source"] == channel]
            if not row.empty:
                ids: list[int] = row["id"].values[0]
                values = row["value"].values[0]
                assert type(ids) == list, f"type(ids): {type(ids)}"
                if len(ids) > 0:
                    assert type(ids[0]) == int, f"type(ids[0]): {type(ids[0])}"
                assert type(values) == list, f"type(values): {type(values)}"
                if len(values) != num_users:
                    print(f"WARNING: len(ids)={len(ids)} in df_list not equal to num_users={num_users}!")
                for id, value in zip(ids, values):
                    user_idx = id - 1
                    if value >= 0:
                        previous_row[user_idx] = value
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
            channel: str = row["source"]
            if name in PREVIOUS_ENTRIES and channel in PREVIOUS_ENTRIES[name]:
                previous_values = PREVIOUS_ENTRIES[name][channel]
                ids = row["id"]
                values = row["value"]
                assert type(ids) == list, f"type(ids): {type(ids)}"
                assert type(values) == list, f"type(values): {type(values)}"
                assert type(ids[0]) == int, f"type(ids[0]): {type(ids[0])}"
                for id, value in zip(ids, values):
                    user = id - 1
                    if value >= 0:
                        previous_values[user] = value
                    # FIXME: toggle fix for rate or traffic ratio being absent
                    if "rate" in name or "traffic_ratio" in name:
                        previous_values[user] = 0
