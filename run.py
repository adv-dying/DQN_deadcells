from model import DQN
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from lib import GetScreen, Actions, env, GetHp

import copy

from tensorboardX import SummaryWriter
import time
import random
import pickle
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True


GAMMA = 0.99

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, dones, next_state = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_state),
        )


class Agent:
    def __init__(self, exp_buffer):
        self.buffer = exp_buffer
        self.get_screen = GetScreen.GetScreen()
        self.total_rewards = 0.0
        self.env = env.env()
        self.hpgetter = GetHp.Hp_getter()
        self._reset()

    def _reset(self):
        state = self.env._reset()
        self.shadow = np.stack((state, state, state, state), axis=0)
        self.bosshp = self.hpgetter.get_boss_hp()
        self.playerhp = self.hpgetter.get_self_hp()

    def play_step(self, net, epsilon, device="cuda"):
        done_reward = None

        if np.random.random() < epsilon:
            action = random.randint(0, 6)
        else:
            state = torch.tensor(self.shadow).to(device)
            q_val_v = net(state)
            _, act_v = torch.max(q_val_v, dim=1)
            action = int(act_v[0].item())

        # [Shield, Roll, Attack, Jump, Move_Left, Move_Right, Nothing]
        print(action)
        new_state, reward, is_done, self.playerhp, self.bosshp = self.env.step(
            action, self.playerhp, self.bosshp
        )
        self.total_rewards += reward
        new_shadow = np.append(
            new_state[np.newaxis, :], self.shadow[:3, :, :, :], axis=0
        )
        exp = Experience(self.shadow, action, reward, is_done, new_shadow)
        self.buffer.append(exp)

        self.shadow = new_shadow

        if is_done:
            Actions.Nothing()
            done_reward = self.total_rewards
            time.sleep(6)
            self._reset()

        return done_reward

    def cal_loss(self, batch, net, tgt_net, device="cuda"):
        state, action, reward, done, next_state = batch

        state_v = torch.tensor(state).to(device)
        next_state_v = torch.tensor(next_state).to(device)
        action_v = torch.tensor(action).to(device)
        reward_v = torch.tensor(reward).to(device)
        done_mask = torch.ByteTensor(done).to(device)

        state_action_value = (
            net(state_v).gather(1, action_v.unsqueeze(-1).type(torch.long)).squeeze(-1)
        )

        next_state_value = tgt_net(next_state_v).max(1)[0]
        next_state_value[done_mask] = 0.0
        next_state_value = next_state_value.detach()
        expected_state_action = next_state_value * GAMMA + reward_v
        return nn.MSELoss()(state_action_value, expected_state_action)

    def get_time(self):
        return self.ts1


buffer = ExperienceBuffer(capacity=1500)
device = "cuda"
agent = Agent(buffer)
net = DQN(7).to(device)
net.load_state_dict(torch.load("./checkpoints/best_model.pt"))
epsilon = 2

while True:
    reward = agent.play_step(net, epsilon, device)
