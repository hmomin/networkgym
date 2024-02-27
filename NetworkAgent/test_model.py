import os
import pickle
import torch
from pprint import pprint

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    script_dir,
    "ptd3",
    "saved",
    "sys_default_norm_utility_moved",
    "sys_default_norm_utility_PTD3_beta_3.0_alpha_0.0_step_0010000.Actor",
)
with open(model_path, "rb") as model_file:
    model = torch.load(model_file, map_location="cuda:0")
    print(model)
