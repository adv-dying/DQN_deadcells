import math
import pygetwindow
from model import DQN_action, DQN_move
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from lib import GetScreen, Actions, env, GetHp
import pickle

from torch.utils.tensorboard import SummaryWriter
import time
import pickle
import keyboard
import os


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 10000

BETA_START = 0.4
BETA_END = 1.0
BETA_DECAY = 10000

TAU = 0.001
LR = 1e-4
MIN_PROB = 0.01
CAPACITY = 10000
ALPHA = 0.6

device = 'cuda'

Experience = collections.namedtuple(
    "Experience", field_names=["state", "move", "action", "reward", "done", "new_state", "TD_move", "TD_action", "boss", "new_boss", "player", "new_player"]
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, beta, sample_target):
        if sample_target == 'move':
            priorities = np.array([(experience.TD_move +
                                    MIN_PROB)**ALPHA for experience in self.buffer])
        elif sample_target == 'action':
            priorities = np.array([(experience.TD_action +
                                    MIN_PROB)**ALPHA for experience in self.buffer])
        else:
            raise KeyError(f'wrong input:{sample_target}')
        priorities = priorities/np.sum(priorities)
        indices = np.random.choice(
            len(self.buffer), batch_size, replace=True, p=priorities)
        state, move, action, reward, dones, next_state, td_m, td_a, boss, new_boss, player, new_player = zip(
            *[self.buffer[idx] for idx in indices]
        )
        weight = (
            1/(self.__len__()*np.array([priorities[i] for i in indices], dtype=np.float32)))**beta
        weight /= max(weight)

        return ((
            torch.stack(state),
            np.array(move),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(dones, dtype=np.bool8),
            torch.stack(next_state),
            torch.tensor(boss).unsqueeze(-1),
            torch.tensor(new_boss).unsqueeze(-1),
            torch.tensor(player).unsqueeze(-1),
            torch.tensor(new_player).unsqueeze(-1)),
            indices,
            torch.from_numpy(weight).to(device)
        )

    def update_td_move(self, indice, td_m, a=0.7):
        for td_idx, idx in enumerate(indice):
            self.buffer[idx] = self.buffer[idx]._replace(TD_move=td_m[td_idx])

    def update_td_action(self, indice, td_a, a=0.7):
        for td_idx, idx in enumerate(indice):
            self.buffer[idx] = self.buffer[idx]._replace(
                TD_action=td_a[td_idx]*a)

    def max_td_move(self):
        if self.__len__() == 0:
            return 1.0
        else:
            return max([i.TD_move for i in self.buffer])

    def max_td_action(self):
        if self.__len__() == 0:
            return 1.0
        else:
            return max([i.TD_action for i in self.buffer])


move_net = DQN_move(3).to(device)
move_tgt_net = DQN_move(3).to(device)

action_net = DQN_action(5).to(device)
action_tgt_net = DQN_action(5).to(device)

# load the model
if os.path.isfile("./checkpoints/best_move_model.pt") and os.path.isfile("./checkpoints/best_action_model.pt"):
    move_net.load_state_dict(torch.load("./checkpoints/best_move_model.pt"))
    action_net.load_state_dict(torch.load(
        "./checkpoints/best_action_model.pt"))
    move_tgt_net.load_state_dict(move_net.state_dict())
    action_tgt_net.load_state_dict(action_net.state_dict())
    print("load model")
if os.path.isfile("./checkpoints/frame.npy") and os.path.isfile("./checkpoints/total_rewards.npy") and os.path.isfile("./checkpoints/best_mean.npy"):
    frame_idx = int(np.load("./checkpoints/frame.npy"))
    total_rewards = np.load("./checkpoints/total_rewards.npy")
    total_rewards = total_rewards.tolist()
    best_mean = float(
        np.load("./checkpoints/best_mean.npy"))
    print(frame_idx)
else:
    # if not, set epsilon
    frame_idx = 0
    total_rewards = []
    best_mean = None
    print("new run")

action_optimizer = optim.Adam(action_net.parameters(), lr=LR, amsgrad=True)
move_optimizer = optim.Adam(move_net.parameters(), lr=LR, amsgrad=True)
preframe_idx = frame_idx


# try to load the before buffer
if os.path.isfile("./checkpoints/buffer.pickle"):
    with open("./checkpoints/buffer.pickle", "rb") as f:
        buffer = pickle.load(f)
    print("load buffer")
else:
    buffer = ExperienceBuffer(capacity=CAPACITY)
    print("new buffer")


# self.hp=15482
# boss.hp=234930
# 15 7 7


class Agent:
    def __init__(self, exp_buffer):
        self.buffer = exp_buffer
        self.get_screen = GetScreen.GetScreen()
        self.total_rewards = 0.0
        self.env = env.env()
        self.hpgetter = GetHp.Hp_getter()
        self.criterion = nn.SmoothL1Loss(reduction='none')
        # self._reset()

    def _reset(self):

        self.state = self.get_screen.grab()
        self.total_rewards = 0.0

        self.env._reset()
        self.bosshp = self.hpgetter.get_boss_hp()
        self.playerhp = self.hpgetter.get_self_hp()
        self.normal_boss_hp = self.bosshp/234930
        self.normal_player_hp = self.playerhp/15482

    def play_step(self, move_net, action_net, device="cuda"):
        done_reward = None

        a_q_val_v = action_net(self.state.unsqueeze(
            0), torch.tensor([self.normal_boss_hp]).unsqueeze(0).to(device), torch.tensor([self.normal_player_hp]).unsqueeze(0).to(device))
        _, act_v = torch.max(a_q_val_v, dim=1)
        action = int(act_v[0].item())

        one_hot_a_t = torch.zeros(5).to(device)
        one_hot_a_t[action] = 1
        m_q_val_v = move_net(self.state.unsqueeze(0),
                             one_hot_a_t.unsqueeze(0))

        _, move_v = torch.max(m_q_val_v, dim=1)

        move = int(move_v[0].item())

        reward, is_done, new_playerhp, new_bosshp, player_damaged, boss_damaged = self.env.step(
            move, action, self.playerhp, self.bosshp
        )

        print(f'reward:{reward:.2f},move:{move},action:{action}, is_done:{is_done}, player_hp:{self.playerhp}, boss_hp:{self.bosshp}              ', end='\r')
        # sys.stdout.flush()
        new_state = self.get_screen.grab()
        self.total_rewards += reward

        cur_max_m = self.buffer.max_td_move()
        cur_max_a = self.buffer.max_td_action()

        normal_new_bosshp = new_bosshp/234930
        normal_new_playerhp = new_playerhp/15482
        exp = Experience(self.state, move, action,
                         reward, is_done, new_state, cur_max_m, cur_max_a, self.normal_boss_hp, normal_new_bosshp, self.normal_player_hp, normal_new_playerhp)
        self.buffer.append(exp)

        if is_done:
            Actions.Nothing()
            done_reward = self.total_rewards


        self.state = new_state

        self.playerhp = new_playerhp
        self.normal_player_hp = normal_new_playerhp

        self.bosshp = new_bosshp
        self.normal_boss_hp = normal_new_bosshp

        return done_reward

    
if __name__ == '__main__':

    win = pygetwindow.getWindowsWithTitle('Dead Cells')[0]
    win.size = (960, 540)

    agent = Agent(buffer)

    MAX_FRAMES = 1000000

    done_reward = None

    time_start = time.time()
    agent._reset()

    while frame_idx < MAX_FRAMES:
        frame_idx += 1
        epsilon = 0
        if done_reward is not None:

            if done_reward == 100:
                break
            done_reward = done_reward
            total_rewards.append(done_reward)
            mean_reward = np.mean(total_rewards[-100:])

            print(
                "\nlenbuffer:%d,frame:%d game:%d, reward:%.3f,mean reward: %.3f, eps:%.2f,frame/sec:%.2f"
                % (len(buffer), frame_idx, len(total_rewards), done_reward, mean_reward, epsilon, (frame_idx-preframe_idx)/(time.time()-time_start))
            )


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
        # press q to quit
        if keyboard.is_pressed('q'):
            break
        # press p to pause
        if keyboard.is_pressed('p'):
            print('\npause')
            time.sleep(5)
            while True:
                if keyboard.is_pressed('p'):
                    print('reset')
                    time.sleep(1)
                    break

            # agent._reset()
        done_reward = agent.play_step(move_net, action_net, device)

