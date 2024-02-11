import os
import pickle
import torch.nn as nn

FOLDER_NAME = "saved"


def load_model(model_path: str) -> nn.Module:
    with open(model_path, "rb") as model:
        actor = pickle.load(model)
    return actor


def save_model(model: nn.Module, model_path: str) -> None:
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def move_model_to_device(model: nn.Module, device: str) -> None:
    print(f"Original device: " + str(model.device))
    print(f"Original device on cuda: " + str(next(model.parameters()).get_device()))
    model.to(device)
    model.device = device
    print(f"New device: " + str(model.device))
    print(f"New device on cuda: " + str(next(model.parameters()).get_device()))


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, FOLDER_NAME)
    model_basepaths = os.listdir(models_dir)
    for model_basepath in model_basepaths:
        if ".Actor" not in model_basepath:
            continue
        model_path = os.path.join(models_dir, model_basepath)
        model = load_model(model_path)
        move_model_to_device(model, "cuda:0")
        save_model(model, model_path)
    print("DONE")


if __name__ == "__main__":
    main()
