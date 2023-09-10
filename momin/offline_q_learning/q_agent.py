import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
from copy import deepcopy
from gym.core import Env
from momin.buffer import Buffer
from momin.offline_q_learning.mlp import Network


# FIXME: a lot of this has to be reworked!!!


class OfflineQAgent:
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
        # FIXME: ensure this is the right size!
        self.observation_dim = 13 * 4
        # FIXME: ensure this is the right size!
        self.action_dim = 4
        self.gamma = gamma
        # check if the save_folder path exists
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        # initialize the Q-function approximator
        self.save_name = os.path.join(save_folder, "q_function.pth")
        self.q_function = (
            pickle.load(open(self.save_name, "rb"))
            if should_load and os.path.exists(self.save_name)
            else Network(
                [self.observation_dim + self.action_dim, 256, 256, 1],
                nn.Identity,
                learning_rate,
                self.device,
            )
        )

    def update(
        self,
        miniBatchSize: int,
        trainingSigma: float,
        trainingClip: float,
        updatePolicy: bool,
    ):
        # randomly sample a mini-batch from the replay buffer
        miniBatch = self.buffer.getMiniBatch(miniBatchSize)
        # create tensors to start generating computational graph
        states = T.tensor(miniBatch["states"], requires_grad=True, device=self.device)
        actions = T.tensor(miniBatch["actions"], requires_grad=True, device=self.device)
        rewards = T.tensor(miniBatch["rewards"], requires_grad=True, device=self.device)
        nextStates = T.tensor(
            miniBatch["nextStates"], requires_grad=True, device=self.device
        )
        dones = T.tensor(miniBatch["doneFlags"], requires_grad=True, device=self.device)
        # compute the targets
        targets = self.computeTargets(
            rewards, nextStates, dones, trainingSigma, trainingClip
        )
        # do a single step on each critic network
        Q1Loss = self.computeQLoss(self.critic1, states, actions, targets)
        self.critic1.gradientDescentStep(Q1Loss, True)
        Q2Loss = self.computeQLoss(self.critic2, states, actions, targets)
        self.critic2.gradientDescentStep(Q2Loss)
        if updatePolicy:
            # do a single step on the actor network
            policyLoss = self.computePolicyLoss(states)
            self.actor.gradientDescentStep(policyLoss)
            # update target networks
            self.updateTargetNetwork(self.targetActor, self.actor)
            self.updateTargetNetwork(self.targetCritic1, self.critic1)
            self.updateTargetNetwork(self.targetCritic2, self.critic2)

    def computeTargets(
        self,
        rewards: T.Tensor,
        nextStates: T.Tensor,
        dones: T.Tensor,
        trainingSigma: float,
        trainingClip: float,
    ) -> T.Tensor:
        targetActions = self.targetActor.forward(nextStates.float())
        # create additive noise for target actions
        noise = np.random.normal(0, trainingSigma, targetActions.shape)
        clippedNoise = T.tensor(
            np.clip(noise, -trainingClip, +trainingClip), device=self.device
        )
        targetActions = T.clip(targetActions + clippedNoise, -1, +1)
        # compute targets
        targetQ1Values = T.squeeze(
            self.targetCritic1.forward(T.hstack([nextStates, targetActions]).float())
        )
        targetQ2Values = T.squeeze(
            self.targetCritic2.forward(T.hstack([nextStates, targetActions]).float())
        )
        targetQValues = T.minimum(targetQ1Values, targetQ2Values)
        return rewards + self.gamma * (1 - dones) * targetQValues

    def computeQLoss(
        self, network: Network, states: T.Tensor, actions: T.Tensor, targets: T.Tensor
    ) -> T.Tensor:
        # compute the MSE of the Q function with respect to the targets
        QValues = T.squeeze(network.forward(T.hstack([states, actions]).float()))
        return T.square(QValues - targets).mean()

    def updateTargetNetwork(self, targetNetwork: Network, network: Network):
        with T.no_grad():
            for targetParameter, parameter in zip(
                targetNetwork.parameters(), network.parameters()
            ):
                targetParameter.mul_(1 - self.tau)
                targetParameter.add_(self.tau * parameter)

    def save(self):
        name = self.envName
        pickle.dump(self.buffer, open(name + "Replay", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))
        pickle.dump(self.critic1, open(name + "Critic1", "wb"))
        pickle.dump(self.critic2, open(name + "Critic2", "wb"))
        pickle.dump(self.targetActor, open(name + "TargetActor", "wb"))
        pickle.dump(self.targetCritic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.targetCritic2, open(name + "TargetCritic2", "wb"))
