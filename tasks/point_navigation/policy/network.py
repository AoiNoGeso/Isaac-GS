import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class CNNEncoder(nn.Module):
    def __init__(self, img_size: int = 84):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, 3, img_size, img_size)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, 256), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class GoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ELU())

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.fc(g)


class PointNavActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        self._img_size = img_size
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                               min_log_std=-20, max_log_std=2)
        self.cnn = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb, goal = obs["rgb"], obs["goal"]
        else:
            goal = obs[:, :2]
            rgb = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        feat = torch.cat([self.cnn(rgb), self.goal(goal)], dim=1)
        mean = self.mean_layer(self.mlp(feat))
        return mean, {"log_std": self.log_std_param.expand_as(mean)}


class PointNavCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        self._img_size = img_size
        DeterministicMixin.__init__(self, clip_actions=False)
        self.cnn = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb, goal = obs["rgb"], obs["goal"]
        else:
            goal = obs[:, :2]
            rgb = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        return self.mlp(torch.cat([self.cnn(rgb), self.goal(goal)], dim=1)), {}


class SACPointNavActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        self._img_size = img_size
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-5, max_log_std=2)
        self.cnn = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_layer = nn.Linear(64, self.num_actions)

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb, goal = obs["rgb"], obs["goal"]
        else:
            goal = obs[:, :2]
            rgb = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        hidden = self.mlp(torch.cat([self.cnn(rgb), self.goal(goal)], dim=1))
        return self.mean_layer(hidden), {"log_std": self.log_std_layer(hidden)}

    def random_act(self, inputs: dict, *, role: str = ""):
        batch_size = inputs["observations"].shape[0]
        actions = torch.rand(batch_size, self.num_actions, device=self.device) * 2.0 - 1.0
        return actions, {}

    def act(self, inputs: dict, *, role: str = ""):
        mean, outputs = self.compute(inputs, role)
        log_std = torch.clamp(outputs["log_std"], self._g_min_log_std, self._g_max_log_std)
        outputs["log_std"] = log_std

        dist = torch.distributions.Normal(mean, log_std.exp())
        self._g_distribution = dist

        u = dist.rsample()
        a = torch.tanh(u)

        taken = inputs.get("taken_actions")
        u_ref = taken if taken is not None else u
        log_prob = dist.log_prob(u_ref).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = torch.tanh(mean)
        return a, outputs


class SACPointNavCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, img_size: int = 84):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        self._img_size = img_size
        DeterministicMixin.__init__(self, clip_actions=False)
        self.cnn = CNNEncoder(img_size=img_size)
        self.goal = GoalEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(256 + 32 + action_space.shape[0], 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        if isinstance(obs, dict):
            rgb, goal = obs["rgb"], obs["goal"]
        else:
            goal = obs[:, :2]
            rgb = obs[:, 2:].view(-1, 3, self._img_size, self._img_size)
        feat = torch.cat([self.cnn(rgb), self.goal(goal), inputs["taken_actions"]], dim=1)
        return self.mlp(feat), {}
