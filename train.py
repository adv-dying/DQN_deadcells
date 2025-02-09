# %%
import math
import pygetwindow
from model import DQN
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from lib import GetScreen, Actions, env, GetHp
import pickle
import copy

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time
import random
import pickle

import torch.backends.cudnn as cudnn

import keyboard

cudnn.deterministic = True

cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# %%
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 10000
TAU = 0.001

LR = 5e-4

device = 'cuda'
# %%
writepath = 'runs/dueling_double_DQN'+'_batch_' + \
    str(BATCH_SIZE)+'_EPS_DECAY_'+str(EPS_DECAY) + \
    '_TAU_'+str(TAU)+'reward_cap_action_move_seperate'
writer = SummaryWriter(log_dir=writepath)
# %%

Experience = collections.namedtuple(
    "Experience", field_names=["state", "move", "action", "reward", "done", "new_state"]
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
        state, move, action, reward, dones, next_state = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            torch.stack(state),
            np.array(move),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(dones, dtype=np.bool8),
            torch.stack(next_state),
        )


# %%
move_net = DQN(3).to(device)
move_tgt_net = DQN(3).to(device)

action_net = DQN(5).to(device)
action_tgt_net = DQN(5).to(device)

# load the model
try:
    move_net.load_state_dict(torch.load("./checkpoints/best_move_model.pt"))
    action_net.load_state_dict(torch.load(
        "./checkpoints/best_action_model.pt"))

    frame_idx = int(np.load("./checkpoints/frame.npy"))
    total_rewards = np.load("./checkpoints/total_rewards.npy")
    total_rewards = total_rewards.tolist()

    move_tgt_net.load_state_dict(move_net.state_dict())
    action_tgt_net.load_state_dict(action_net.state_dict())

    print(frame_idx)
    print("load model")
except:
    # if not, set epsilon
    frame_idx = 0
    total_rewards = []
    print("new model")

action_optimizer = optim.Adam(action_net.parameters(), lr=LR, amsgrad=True)
move_optimizer = optim.Adam(move_net.parameters(), lr=LR, amsgrad=True)
preframe_idx = frame_idx


# %%
# try to load the before buffer
try:
    with open("./checkpoints/buffer.pickle", "rb") as f:
        buffer = pickle.load(f)
    print("load buffer")
except:
    buffer = ExperienceBuffer(capacity=10000)
    print("new buffer")

# %%
# self.hp=15482
# boss.hp=215249
# 15 7 7


class Agent:
    def __init__(self, exp_buffer):
        self.buffer = exp_buffer
        self.get_screen = GetScreen.GetScreen()
        self.total_rewards = 0.0
        self.env = env.env()
        self.hpgetter = GetHp.Hp_getter()
        self.criterion = nn.SmoothL1Loss()

        # self.min_reward = -0.01
        # self.max_reward = 0.01
        # self._reset()

    def _reset(self):
        self.env._reset()
        self.state = self.get_screen.grab()
        self.total_rewards = 0.0
        self.bosshp = self.hpgetter.get_boss_hp()
        self.playerhp = self.hpgetter.get_self_hp()

    def play_step(self, move_net, action_net, epsilon, device="cuda"):
        done_reward = None

        if np.random.random() < epsilon:
            move = random.randint(0, 2)
            action = random.randint(0, 4)

        else:
            m_q_val_v = move_net(self.state.unsqueeze(0))
            a_q_val_v = action_net(self.state.unsqueeze(0))
            _, move_v = torch.max(m_q_val_v, dim=1)
            _, act_v = torch.max(a_q_val_v, dim=1)
            move = int(move_v[0].item())
            action = int(act_v[0].item())

        # Actions = [Attack,Shield, Roll, Jump, Move_Left, Move_Right, Nothing]
        reward, is_done, self.playerhp, self.bosshp = self.env.step(
            move, action, self.playerhp, self.bosshp
        )
        # self.min_reward = min(self.min_reward, reward)
        # self.max_reward = max(self.max_reward, reward)
        if reward != 0:
            print(
                f"move:{move},action:{action},reward:{reward:.2f},bosshp:{self.bosshp},selfhp:{self.playerhp} ")
        new_state = self.get_screen.grab()
        self.total_rewards += reward

        # normalize_reward = (reward - self.min_reward) / \
        #     (self.max_reward - self.min_reward)
        exp = Experience(self.state, move, action,
                         reward, is_done, new_state)
        self.buffer.append(exp)

        self.state = new_state

        if is_done:
            Actions.Nothing()
            done_reward = self.total_rewards
            # print("reward:%.2f" % (done_reward), end='\r')

        return done_reward

    def cal_loss(self, batch, move_net, move_tgt_net, action_net, action_tgt_net, device="cuda"):
        state, move, action, reward, done, next_state = batch

        state_v = state.to(device)
        move_v = torch.tensor(move, dtype=torch.int64).to(device)
        action_v = torch.tensor(action, dtype=torch.int64).to(device)
        reward_v = torch.tensor(reward).to(device)
        next_state_v = next_state.to(device)

        # state_action_value = (
        #     net(state_v).gather(
        #         1, action_v.unsqueeze(-1).type(torch.long)).squeeze(-1)
        # )
        state_move_value = (move_net(state_v).gather(
            1, move_v.unsqueeze(-1))).squeeze(-1)
        state_action_value = (action_net(state_v).gather(
            1, action_v.unsqueeze(-1))).squeeze(-1)

        with torch.no_grad():
            argmax_m = move_net(next_state_v).argmax(dim=1).unsqueeze(-1)
            argmax_a = action_net(next_state_v).argmax(dim=1).unsqueeze(-1)
            next_state_value_m = move_tgt_net(
                next_state_v).gather(1, argmax_m).squeeze(1)
            next_state_value_a = action_tgt_net(
                next_state_v).gather(1, argmax_a).squeeze(1)
            # next_state_value = tgt_net(next_state_v).max(1)[0]
            next_state_value_m[done] = 0.0
            next_state_value_a[done] = 0.0
        expected_state_move = next_state_value_m*GAMMA+reward_v
        expected_state_action = next_state_value_a * GAMMA + reward_v

        return self.criterion(state_move_value, expected_state_move), self.criterion(state_action_value, expected_state_action)

    def optimize_model(self, idx):
        if buffer.__len__() < BATCH_SIZE:
            return
        batch = buffer.sample(BATCH_SIZE)
        loss_m, loss_a = self.cal_loss(
            batch, move_net, move_tgt_net, action_net, action_tgt_net)
        writer.add_scalar("loss/lossm", loss_m, idx)
        writer.add_scalar("loss/lossa", loss_a, idx)
        move_optimizer.zero_grad()
        loss_m.backward()
        torch.nn.utils.clip_grad_value_(move_net.parameters(), 100)
        move_optimizer.step()

        action_optimizer.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_value_(action_net.parameters(), 100)
        action_optimizer.step()

        # sync the tgt_net hard
        # if idx % 100 == 0:
        #     print('sync', end='\r')
        #     tgt_net.load_state_dict(net.state_dict())
        # change to soft sync
        move_tgt_net_state_dict = move_tgt_net.state_dict()
        move_net_state_dict = move_net.state_dict()
        for key in move_net_state_dict:
            move_tgt_net_state_dict[key] = move_net_state_dict[key] * \
                TAU + move_tgt_net_state_dict[key]*(1-TAU)
        move_tgt_net.load_state_dict(move_tgt_net_state_dict)

        action_tgt_net_state_dict = action_tgt_net.state_dict()
        action_net_state_dict = action_net.state_dict()
        for key in action_net_state_dict:
            action_tgt_net_state_dict[key] = action_net_state_dict[key] * \
                TAU + action_tgt_net_state_dict[key]*(1-TAU)
        action_tgt_net.load_state_dict(action_tgt_net_state_dict)


if __name__ == '__main__':
    # %%
    win = pygetwindow.getWindowsWithTitle('Dead Cells')[0]
    win.size = (960, 540)

    # %%
    agent = Agent(buffer)
    # agent.get_screen.show()
    # best_mean_reward = None
    pre_save = frame_idx//10000
    MAX_FRAMES = 1000000
    # %%

    done_reward = None

    # numbers of game
    time_start = time.time()
    agent._reset()

    while frame_idx < MAX_FRAMES:
        frame_idx += 1
        epsilon = EPS_END + (EPS_START-EPS_END) * \
            math.exp(-1. * frame_idx / EPS_DECAY)

        if done_reward is not None:
            # one loop ends
            # avoid Deadcells break
            if done_reward == 10:
                break

            total_rewards.append(done_reward)
            mean_reward = np.mean(total_rewards[-100:])

            print(
                "lenbuffer:%d,frame:%d game:%d, reward:%.3f,mean reward: %.3f, eps:%.2f,frame/sec:%.2f"
                % (len(buffer), frame_idx, len(total_rewards), done_reward, mean_reward, epsilon, (frame_idx-preframe_idx)/(time.time()-time_start))
            )
            # for idx in range(preframe_idx, frame_idx):
            #     agent.optimize_model(idx)
            # time.sleep(4)

            #     print(x, end='\r')
            # print()
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("reward/reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward/reward", done_reward, frame_idx)

            # save model
            if frame_idx//10000 > pre_save:
                pre_save += 1
                torch.save(move_net.state_dict(),
                           "./checkpoints/best_move_model.pt")
                torch.save(action_net.state_dict(),
                           "./checkpoints/best_action_model.pt")
                np.save("./checkpoints/frame.npy", frame_idx)
                np.save("./checkpoints/total_rewards.npy", total_rewards)
                # save buffer
                with open("./checkpoints/buffer.pickle", "wb") as f:
                    pickle.dump(copy.deepcopy(buffer), f)
                # if best_mean_reward is not None:
                #     print(
                #         "Best mean reward updated %.3f -> %.3f, model saved"
                #         % (best_mean_reward, mean_reward)
                #     )
                # best_mean_reward = mean_reward

            # reset game
            agent._reset()

            # if for some random reason that agent do not enter the boss region
            if not agent.hpgetter.get_boss_hp():
                time.sleep(1)
                Actions.Move_Right()
                time.sleep(8)
                Actions.Nothing()
                Actions.Move_Left()
                time.sleep(5.5)
                Actions.Nothing()
                agent._reset()
            time_start = time.time()
            preframe_idx = frame_idx
        # play a step
        if keyboard.is_pressed('q'):
            break
        done_reward = agent.play_step(move_net, action_net, epsilon, device)
        agent.optimize_model(frame_idx)

    writer.close()

    # # %%
    # torch.save(net.state_dict(), "./checkpoints/best_model.pt")
    # np.save("./checkpoints/frame.npy", frame_idx)
    # np.save("./checkpoints/total_rewards.npy", total_rewards)
    # with open("./checkpoints/buffer.pickle", "wb") as f:
    #     pickle.dump(copy.deepcopy(buffer), f)
