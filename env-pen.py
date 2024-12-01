import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
import copy
warnings.filterwarnings("ignore")

__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class BehaviorCloningNN(nn.Module):
    def __init__(self):
        super(BehaviorCloningNN, self).__init__()
        self.fc1 = nn.Linear(3, 9)  # 输入3维，隐藏层9个节点
        self.fc2 = nn.Linear(9, 1)  # 输出1维
        self.output = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)
        x = self.output(x) * 2
        return x


class CustomLSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size1=16, hidden_size2=256, output_size=3):
        super(CustomLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x


class PendulumEnv_guo(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.policy = BehaviorCloningNN()
        self.policy.load_state_dict(
            torch.load('/Users/ictnie/Desktop/rl-baselines3-zoo-master/g/20241201_164821/bc_model.pth'))
        self.trans = CustomLSTMModel(input_size=4, output_size=3)
        self.trans.load_state_dict(torch.load('/Users/ictnie/Desktop/rl-baselines3-zoo-master/g/20241201_164821/transition.pth'))

    ## 两个动作，一个是 是否攻击，一个是 扰动🤔
    def step(self, delt):
        with torch.no_grad():
            obs1 = self.state
            obs1 = torch.tensor(obs1, dtype=torch.float32)
            a1 = self.policy(obs1)

            obs1 += delt

            a2 = self.policy(obs1)
            obs2 = torch.tensor([[obs1[0], obs1[1], obs1[2], a2[0]]], dtype=torch.float32)
            obs_n = self.trans(obs2)
            self.state = [obs_n[0][i] for i in range(len(obs_n[0]))]
            cost = abs(a1 - a2)[0]

        return self._get_state(), cost, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.array([DEFAULT_X, DEFAULT_Y])
        low = -high  # We enforce symmetric limits.
        p = self.np_random.uniform(low=low, high=high)
        theta, thetadot = p
        self.state = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        return self._get_state(), {}

    def _get_state(self):
        state=self.state
        return np.array([i for i in state], dtype=np.float32)
