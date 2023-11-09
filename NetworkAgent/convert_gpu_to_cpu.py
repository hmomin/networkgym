import os
import pickle

MODEL_NAME = "SAC_deterministic_walk_bc.10000.64.0.250.not_normalized.Actor"
SAVE_NAME = "NEW_SAC_deterministic_walk_bc.10000.64.0.250.not_normalized.Actor"


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_loc = os.path.join(script_dir, "stable-baselines3", "models", MODEL_NAME)
    actor = pickle.load(open(model_loc, "rb"))
    return actor


def save_model(model) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "stable-baselines3", "models", SAVE_NAME)
    pickle.dump(model, open(model_dir, "wb"))


def main() -> None:
    model = load_model()
    print(f"Original device: " + str(model.device))
    print(f"Original device on cuda: " + str(next(model.parameters()).is_cuda))
    model.to("cpu")
    model.device = "cpu"
    print(f"New device: " + str(model.device))
    print(f"New device on cuda: " + str(next(model.parameters()).is_cuda))
    save_model(model)
    print("DONE")


if __name__ == "__main__":
    main()
