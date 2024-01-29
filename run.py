from model import DQN
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from lib import GetScreen, Actions, env, GetHelper
from lib.SendKey import PressKey, ReleaseKey
import time
import random
import pickle
import copy
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import threading
import sys

cudnn.deterministic = True

cudnn.benchmark = True
torch.backends.cudnn.enabled = True

UP_ARROW = 0x26
R = 0x52

GAMMA = 0.99
MAX_EPISODE = int(input("play_turn:"))
ESC = 0x1B

epsilon_final = 0.02
epsilonDecay = 5 * 10**4
replay_min_size = 500
batch_size = 24
device = "cuda"
epsilon_start = 1.0

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
            np.array(action, dtype=np.int64),
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
        self.getter = GetHelper.getter()
        self._reset()
        print("check selfHP:%d" % self.playerhp)
        print("check bossHP:%d" % self.bosshp)
        print("check selfpos:%d" % self.getter.self_pos())
        print("check bosspos:%d" % self.getter.boss_pos())

    def _reset(self):
        state = self.env._reset()
        self.total_rewards = 0.0
        self.bosshp = self.getter.boss_hp()
        self.playerhp = self.getter.self_hp()
        self.shadow = np.stack((state, state, state, state), axis=0)
        self.parameter = 0

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
        (
            new_state,
            reward,
            is_done,
            self.playerhp,
            self.bosshp,
            self.parameter,
        ) = self.env.step(action, self.playerhp, self.bosshp, self.parameter)
        print(
            "action:%d,reward:%d,bosspos:%d,selfpos:%d,bossHP:%d,selfHP:%d,parameter:%d"
            % (action, reward, self.getter.boss_pos(), self.getter.self_pos(), self.bosshp, self.playerhp,self.parameter),
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
            print("reward:%.2f" % (done_reward))

        return done_reward

    def cal_loss(self, batch, net, tgt_net, device="cuda"):
        state, action, reward, done, next_state = batch

        state_v = torch.tensor(state).to(device)
        next_state_v = torch.tensor(next_state).to(device)
        action_v = torch.tensor(action).to(device)
        reward_v = torch.tensor(reward).to(device)
        done_mask = torch.tensor(done, dtype=torch.bool).to(device)

        state_action_value = net(state_v).gather(1, action_v.unsqueeze(-1)).squeeze(-1)

        next_state_value = tgt_net(next_state_v).max(1)[0]
        next_state_value[done_mask] = 0.0
        next_state_value = next_state_value.detach()
        expected_state_action = next_state_value * GAMMA + reward_v
        return nn.MSELoss()(state_action_value, expected_state_action)


net = DQN(7).to(device)
tgt_net = DQN(7).to(device)

# load the model
try:
    net.load_state_dict(torch.load("./checkpoints/best_model.pt"))
    frame_idx = np.load("./checkpoints/frame.npy")
    tgt_net.load_state_dict(net.state_dict())
    print("frame:%d" % frame_idx)
    print("load model")
except:
    # if not, set epsilonr
    frame_idx = 0
    print("new model")

optimizer = optim.Adam(net.parameters(), lr=0.001)

writer = SummaryWriter(comment="deadcells")

# try to load the before experience
try:
    with open("./checkpoints/buffer.pickle", "rb") as f:
        buffer = pickle.load(f)
    print("load buffer")
except:
    buffer = ExperienceBuffer(capacity=700)
    print("new buffer")

agent = Agent(buffer)
total_rewards = []


reward = None
# numbers of game
time_start = time.time()
best_reward = -(sys.maxsize - 1)
while len(total_rewards) < MAX_EPISODE:
    frame_idx += 1
    epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilonDecay)

    if reward is not None:
        # one loop ends

        # wait for load
        time.sleep(8)

        if reward == 0.0:
            break
        if reward > best_reward:
            best_reward = reward
        # np.save("./checkpoints/frame.npy", frame_idx)

        # save buffer
        # with open("./checkpoints/buffer.pickle", "wb") as f:
        #     pickle.dump(copy.deepcopy(buffer), f)
        # optimize
        # if len(buffer) > replay_min_size:
        #     print("optim")
        #     optimizer.zero_grad()
        #     batch = buffer.sample(batch_size)
        #     loss_t = agent.cal_loss(batch, net, tgt_net)
        #     loss_t.backward()
        #     optimizer.step()

        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-50:])
        print(
            "%d:done %d/%d:game, mean reward: %.3f, best_reward: %.3f eps:%.2f"
            % (
                frame_idx,
                len(total_rewards),
                MAX_EPISODE,
                mean_reward,
                best_reward,
                epsilon,
            )
        )
        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        # save model
        # torch.save(net.state_dict(), "./checkpoints/best_model.pt")

        # step between copy the net to tgt_net.
        if len(total_rewards) % 5 == 0:
            tgt_net.load_state_dict(net.state_dict())
        # reset game

        print("reset")
        agent._reset()

    reward = agent.play_step(net, epsilon, device)


writer.close()
print(time.localtime())
print((time.time() - time_start) / 60)
