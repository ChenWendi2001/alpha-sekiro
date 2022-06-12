from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.common_layers = nn.Sequential(
            nn.Linear(4, 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
        )

        # A head
        self.A_output = nn.Sequential(
            nn.Linear(10, 2)
        )

        # V head
        self.V_output = nn.Sequential(
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.common_layers(x)
        A = self.A_output(x)
        V = self.V_output(x)
        Q = V + A - A.mean(dim=1).view(-1, 1)
        return Q


class D3QN():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")

        self.q_net = VANet().to(self.device)
        # NOTE: target net is used for training
        self.target_net = VANet().to(self.device)

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=0.001)

    def save(self, version):
        checkpoint_dir = "checkpoints"

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        print("save network & optimizer / version({})".format(version))
        torch.save(
            self.target_net.state_dict(),
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_dir, optimizer_dir=None):
        print("load network {}".format(model_dir))

        self.q_net.load_state_dict(torch.load(
            model_dir, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

        if not optimizer_dir is None:
            print("load optimizer {}".format(optimizer_dir))
            self.optimizer.load_state_dict(torch.load(optimizer_dir))

    def softUpdateTarget(self):
        tau = 1
        for t_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            t_param.data = t_param.data * (1.0 - tau) + param.data * tau

    def predict(self, features):
        """[summary]
        NOTE: use encoder to encode state
            before calling predict
        """
        if features.ndim < 2:
            features = np.expand_dims(features, 0)

        features = torch.from_numpy(
            features).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(features)
        return q_values.detach().cpu().numpy()

    def trainStep(self, data_batch: List):
        """[summary]
        Returns:
            loss
        """
        states, rewards, actions, next_states, dones = data_batch
        states = states.float().to(self.device)
        rewards = rewards.view(-1, 1).float().to(self.device)
        actions = actions.view(-1, 1).long().to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.view(-1, 1).float().to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_actions = self.q_net(next_states).argmax(dim=1)
            tq_values = self.target_net(
                next_states).gather(1, max_actions.view(-1, 1))

        q_targets = rewards + (1 - dones) * 0.98 * tq_values
        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.softUpdateTarget()

        return loss.item()
