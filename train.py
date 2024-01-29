from model import DQN
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from lib import GetScreen, Actions, env, GetHelper

import time
import random
import pickle
import copy
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

import sys

cudnn.deterministic = True

cudnn.benchmark = True
torch.backends.cudnn.enabled = True

UP_ARROW = 0x26
R = 0x52

GAMMA = 0.99
MAX_EPISODE = int(input("play_turn:"))
frame = int(input("frame:"))
ESC = 0x1B

replay_min_size = 200
batch_size = 16
device = "cuda:0"
delay = 10**4

Experience = collections.namedtuple("Experience",
                                    field_names=[
                                        "state", "attack", "move",
                                        "hit_reward", "move_reward", "done",
                                        "new_state"
                                    ])


class ExperienceBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, attack, move, hit_reward, move_reward, dones, next_state = zip(
            *[self.buffer[idx] for idx in indices])

        return (
            state,
            np.array(attack, dtype=np.int64),
            np.array(move, dtype=np.int64),
            np.array(hit_reward, dtype=np.float32),
            np.array(move_reward, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            next_state,
        )


class Agent:

    def __init__(self, exp_buffer):
        self.buffer = exp_buffer
        self.get_screen = GetScreen.GetScreen()
        self.total_move_rewards = 0.0
        self.total_hit_rewards = 0.0
        self.env = env.env()
        self.getter = GetHelper.getter()
        self._reset()

        print("check selfHP:%d" % self.playerhp)
        print("check bossHP:%d" % self.bosshp)
        print("check selfpos:%d" % self.getter.self_pos())
        print("check bosspos:%d" % self.getter.boss_pos())

    def _reset(self):
        self.env._reset()
        self.total_hit_rewards = 0.0
        self.total_move_rewards = 0.0
        self.bosshp = self.getter.boss_hp()
        self.playerhp = self.getter.self_hp()
        self.bosspos = self.getter.boss_pos()
        self.playerpos = self.getter.self_pos()

    def play_step(self, attack_net, move_net, episolon, device="cuda:0"):
        done_reward = None
        state = self.get_screen.grab()

        if np.random.random() < episolon:
            move = random.randint(0, 4)
            attack = random.randint(0, 3)

        else:
            state_v = torch.tensor(state).to(device)
            state_v = state_v.unsqueeze(0)

            q_val_a = attack_net(state_v)
            _, a_v = torch.max(q_val_a, dim=1)

            q_val_m = move_net(state_v)
            _, m_v = torch.max(q_val_m, dim=1)

            attack = int(a_v[0].item())
            move = int(m_v[0].item())

        (
            new_state,
            move_reward,
            hit_reward,
            is_done,
            self.playerhp,
            self.bosshp,
        ) = self.env.step(
            attack,
            move,
            self.playerhp,
            self.bosshp,
        )

        self.total_hit_rewards += hit_reward
        self.total_move_rewards += move_reward

        exp = Experience(state, attack, move, hit_reward, move_reward, is_done,
                         new_state)
        self.buffer.append(exp)

        if is_done:
            Actions.Nothing()
            done_reward = [self.total_hit_rewards, self.total_move_rewards]
            print(done_reward)

        return done_reward

    def cal_loss(self,
                 batch,
                 move_net,
                 tgt_move_net,
                 attack_net,
                 tgt_attack_net,
                 device="cuda:0"):
        state, attack, move, hit_reward, move_reward, done, next_state = batch

        # state_v = torch.tensor(state).to(device)
        # next_state_v = torch.tensor(next_state).to(device)
        state_v = torch.tensor([item.cpu().detach().numpy()
                                for item in state]).to(device)
        next_state_v = torch.tensor(
            [item.cpu().detach().numpy() for item in next_state]).to(device)
        attack_v = torch.tensor(attack).to(device)
        move_v = torch.tensor(move).to(device)
        hit_reward_v = torch.tensor(hit_reward).to(device)
        move_reward_v = torch.tensor(move_reward).to(device)
        done_mask = torch.tensor(done, dtype=torch.bool).to(device)

        a_state_action_value = attack_net(state_v).gather(
            1, attack_v.unsqueeze(-1)).squeeze(-1)

        a_next_state_value = tgt_attack_net(next_state_v).max(1)[0]
        a_next_state_value[done_mask] = 0.02
        a_next_state_value = a_next_state_value.detach()
        a_expected_state_action = a_next_state_value * GAMMA + hit_reward_v

        m_state_action_value = move_net(state_v).gather(
            1, move_v.unsqueeze(-1)).squeeze(-1)

        m_next_state_value = tgt_move_net(next_state_v).max(1)[0]
        m_next_state_value[done_mask] = 0.0
        m_next_state_value = m_next_state_value.detach()
        m_expected_state_action = m_next_state_value * GAMMA + move_reward_v
        return nn.MSELoss()(a_state_action_value,
                            a_expected_state_action), nn.MSELoss()(
                                m_state_action_value, m_expected_state_action)


move_net = DQN(5).to(device)
tgt_move_net = DQN(5).to(device)

attack_net = DQN(4).to(device)
tgt_attack_net = DQN(4).to(device)

# load the model
try:
    attack_net.load_state_dict(torch.load("./checkpoints/attack.pt"))
    tgt_attack_net.load_state_dict(attack_net.state_dict())

    move_net.load_state_dict(torch.load("./checkpoints/move.pt"))

    print("load model")
except:
    print("new model")

a_optimizer = optim.Adam(attack_net.parameters(), lr=0.001)
m_optimizer = optim.Adam(move_net.parameters(), lr=0.001)

writer = SummaryWriter(comment="deadcells")

buffer = ExperienceBuffer(capacity=700)

agent = Agent(buffer)
total_rewards = []

reward = None
# numbers of game
time_start = time.time()
best_reward = -(sys.maxsize - 1)
while len(total_rewards) < MAX_EPISODE:
    episolon = max(0.02, 1 - frame / delay)
    frame += 1
    print(1)
    if reward is not None:
        # one loop ends

        # wait for load
        time.sleep(8)

        # save buffer
        # with open("./checkpoints/buffer.pickle", "wb") as f:
        #     pickle.dump(copy.deepcopy(buffer), f)
        # optimize
        if len(buffer) > replay_min_size:
            print("optim")

            a_optimizer.zero_grad()

            batch = buffer.sample(batch_size)

            a_loss_t, m_loss_t = agent.cal_loss(batch, move_net, tgt_move_net,
                                                attack_net, tgt_attack_net)
            a_loss_t.backward()
            a_optimizer.step()

            m_optimizer.zero_grad()

            m_loss_t.backward()
            m_optimizer.step()

        total_rewards.append(reward)

        attack_mean_reward = np.mean([i[0]for i in total_rewards[-50:]])
        move_mean_reward = np.mean([i[1]for i in total_rewards[-50:]])

        print(
            "%d/%d:game, attack mean reward: %.3f,move_mean_reward,%.3f,frame:%.3f ,episolon:%.3f "
            % (
                len(total_rewards),
                MAX_EPISODE,
                attack_mean_reward,
                move_mean_reward,
                frame,
                episolon,
            ))

        writer.add_scalar("attack_reward", reward[0])
        writer.add_scalar("move_reward", reward[1])

        # save model
        torch.save(move_net.state_dict(), "./checkpoints/move.pt")
        torch.save(attack_net.state_dict(), "./checkpoints/attack.pt")
        # step between copy the net to tgt_net.
        if len(total_rewards) % 5 == 0:
            tgt_move_net.load_state_dict(move_net.state_dict())
            tgt_attack_net.load_state_dict(attack_net.state_dict())
        # reset game

        print("reset")
        agent._reset()

    reward = agent.play_step(attack_net, move_net, episolon)

writer.close()
print(time.localtime())
print((time.time() - time_start) / 60)
