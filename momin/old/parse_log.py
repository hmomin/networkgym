# NOTE: used to examine which pieces of the dataframe are missing
# (typically when some link was unused on a step)

import os
from pprint import pprint
from typing import Dict, Tuple

FILENAME = "shape_log.txt"


def get_count_from_line(line: str) -> Tuple[str, int]:
    chunks = line.split(",")
    assert len(chunks) == 2
    chunks[1] = int(chunks[1])
    return tuple(chunks)


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(this_dir, "logs", FILENAME)
    with open(log_path, "r") as log:
        file_content = log.readlines()
    counter_dict: Dict[str, Dict[int, int]] = {}
    for line in file_content:
        if "COUNTER" in line:
            continue
        name, count = get_count_from_line(line)
        if name not in counter_dict:
            counter_dict[name] = {}
        name_dict = counter_dict[name]
        name_dict[count] = name_dict.get(count, 0) + 1
    pprint(counter_dict)


if __name__ == "__main__":
    main()
