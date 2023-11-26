from stable_baselines3 import PPO

# FIXME LOW: this implementation of PPO should allow us to decrease log_std linearly
# may need to revisit later...

class CustomPPO(PPO):
    def train(self) -> None:
        print("")
        self.policy.log_std.requires_grad_(False)