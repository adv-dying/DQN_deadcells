# %%
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
import copy


from torch.utils.tensorboard import SummaryWriter
import time
import random
import pickle
import keyboard
import sys


# %%
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 10000

BETA_START = 0.4
BETA_END = 1.0
BETA_DECAY = 10000

TAU = 0.01
LR = 1e-4
MIN_PROB = 0.01
CAPACITY = 10000
ALPHA = 0.6

device = 'cuda'
# %%
writepath = f'runs/dueling_double_DQN_3fc_ALPHA_{str(ALPHA)}_Beta_{str(BETA_DECAY)}_capacity_{CAPACITY}_batch_{str(BATCH_SIZE)}_EPS_DECAY_{str(EPS_DECAY)}_TAU_{str(TAU)}_LR1e-4/prioritized_replay_buffer_IS+grab(128,128)_linear_td_access_hp_modify_weight'
writer = SummaryWriter(log_dir=writepath)
# %%

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
            self.buffer[idx] = self.buffer[idx]._replace(TD_move=self.buffer[idx].TD_move *
                                                         (1-a)+td_m[td_idx]*a)

    def update_td_action(self, indice, td_a, a=0.7):
        for td_idx, idx in enumerate(indice):
            self.buffer[idx] = self.buffer[idx]._replace(TD_action=self.buffer[idx].TD_action *
                                                         (1-a)+td_a[td_idx]*a)

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


# %%
move_net = DQN_move(3).to(device)
move_tgt_net = DQN_move(3).to(device)

action_net = DQN_action(5).to(device)
action_tgt_net = DQN_action(5).to(device)

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
    buffer = ExperienceBuffer(capacity=CAPACITY)
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
        self.criterion = nn.SmoothL1Loss(reduction='none')
        # self._reset()

    def _reset(self):

        self.state = self.get_screen.grab()
        self.total_rewards = 0.0

        self.env._reset()
        self.bosshp = self.hpgetter.get_boss_hp()
        self.playerhp = self.hpgetter.get_self_hp()
        self.normal_boss_hp = self.bosshp/215249
        self.normal_player_hp = self.playerhp/15482

    def play_step(self, move_net, action_net, epsilon, device="cuda"):
        done_reward = None

        if np.random.random() < epsilon:
            move = random.randint(0, 2)
            action = random.randint(0, 4)

        else:

            a_q_val_v = action_net(self.state.unsqueeze(
                0), torch.tensor([self.normal_player_hp]).unsqueeze(0).to(device), torch.tensor([self.playerhp]).unsqueeze(0).to(device))
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
        if player_damaged != 0:
            writer.add_scalar('damage/player', player_damaged, frame_idx)
        if boss_damaged != 0:
            writer.add_scalar('damage/boss', boss_damaged, frame_idx)
        print(f'reward:{reward:.2f},move:{move},action:{action}, is_done:{is_done}, player_hp:{self.playerhp}, boss_hp:{self.bosshp}              ',end='\r')
        # sys.stdout.flush()
        new_state = self.get_screen.grab()
        self.total_rewards += reward

        cur_max_m = self.buffer.max_td_move()
        cur_max_a = self.buffer.max_td_action()

        normal_new_bosshp = new_bosshp/215249
        normal_new_playerhp = new_playerhp/15482
        exp = Experience(self.state, move, action,
                         reward, is_done, new_state, cur_max_m, cur_max_a, self.normal_boss_hp, normal_new_bosshp, self.normal_player_hp, normal_new_playerhp)
        self.buffer.append(exp)

        self.state = new_state

        self.playerhp = new_playerhp
        self.normal_player_hp = new_playerhp

        self.bosshp = new_bosshp
        self.normal_boss_hp = normal_new_bosshp

        if is_done:
            Actions.Nothing()
            done_reward = self.total_rewards

        return done_reward

    def cal_loss_action(self, batch, net, tgt_net, device="cuda"):
        state, move, action, reward, done, next_state, boss, new_boss, player, new_player = batch

        state_v = state.to(device)
        move_v = torch.tensor(move, dtype=torch.int64).to(device)
        action_v = torch.tensor(action, dtype=torch.int64).to(device)
        reward_v = torch.tensor(reward).to(device)
        next_state_v = next_state.to(device)
        boss_v = torch.tensor(boss).to(device)
        new_boss_v = torch.tensor(new_boss).to(device)
        player_v = torch.tensor(player).to(device)
        new_player_v = torch.tensor(new_player).to(device)
        # state_action_value = (
        #     net(state_v).gather(
        #         1, action_v.unsqueeze(-1).type(torch.long)).squeeze(-1)
        # )

        state_value = (net(state_v, boss_v, player_v).gather(
            1, action_v.unsqueeze(-1))).squeeze(-1)

        with torch.no_grad():

            argmax_a = net(state_v, boss_v, player_v).argmax(
                dim=1).unsqueeze(-1)
            next_state_value = tgt_net(
                next_state_v, new_boss_v, new_player_v).gather(1, argmax_a).squeeze(1)

            next_state_value[done] = 0.0

        expected_state = next_state_value * GAMMA + reward_v
        loss = self.criterion(state_value, expected_state)
        td_error = np.absolute(
            (expected_state-state_value).detach().cpu().numpy())
        return loss, td_error

    def cal_loss_move(self, batch, net, tgt_net, device="cuda"):
        state, move, action, reward, done, next_state, boss, new_boss, player, new_player = batch

        state_v = state.to(device)
        move_v = torch.tensor(move, dtype=torch.int64).to(device)
        reward_v = torch.tensor(reward).to(device)
        next_state_v = next_state.to(device)

        one_hot = torch.zeros((BATCH_SIZE, 5))
        index_tensor = torch.from_numpy(action).long()
        one_hot[torch.arange(BATCH_SIZE), index_tensor] = 1.0
        one_hot = one_hot.to(device)
        state_value = (net(state_v, one_hot).gather(
            1, move_v.unsqueeze(-1))).squeeze(-1)

        with torch.no_grad():

            argmax_a = net(state_v, one_hot).argmax(
                dim=1).unsqueeze(-1)
            next_state_value = tgt_net(
                next_state_v, one_hot).gather(1, argmax_a).squeeze(1)

            next_state_value[done] = 0.0

        expected_state = next_state_value * GAMMA + reward_v
        loss = self.criterion(state_value, expected_state)
        td_error = np.absolute(
            (expected_state-state_value).detach().cpu().numpy())
        return loss, td_error

    def optimize_model(self, frame_idx, beta):
        if buffer.__len__() < BATCH_SIZE:
            return

        batch_m, indice_m, weight_m = self.buffer.sample(
            BATCH_SIZE, beta, 'move')
        batch_a, indice_a, weight_a = self.buffer.sample(
            BATCH_SIZE, beta, 'action')

        loss_m, td_m = self.cal_loss_move(batch_m, move_net, move_tgt_net)
        loss_a, td_a = self.cal_loss_action(
            batch_a, action_net, action_tgt_net)

        loss_m = torch.dot(loss_m, weight_m)
        loss_a = torch.dot(loss_a, weight_a)

        max_td_m = max(td_m)
        max_td_a = max(td_a)

        self.buffer.update_td_move(indice_m, td_m)
        self.buffer.update_td_action(indice_a, td_a)

        writer.add_scalar("loss/lossm", loss_m, frame_idx)
        writer.add_scalar("loss/lossa", loss_a, frame_idx)
        writer.add_scalar("td/max_td_action", max_td_a, frame_idx)
        writer.add_scalar("td/max_td_move", max_td_m, frame_idx)
        writer.add_scalar("weight/move", weight_m[0], frame_idx)
        writer.add_scalar("weight/action", weight_a[0], frame_idx)

        move_optimizer.zero_grad()
        loss_m.backward()
        torch.nn.utils.clip_grad_value_(move_net.parameters(), 5)
        move_optimizer.step()

        action_optimizer.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_value_(action_net.parameters(), 5)
        action_optimizer.step()

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
        beta_frac = frame_idx/BETA_DECAY
        beta = min(beta_frac*BETA_END+(1-beta_frac)*BETA_START, BETA_END)

        if done_reward is not None:
            # one loop ends
            # avoid Deadcells break
            if done_reward == 100:
                break
            done_reward = done_reward
            total_rewards.append(done_reward)
            mean_reward = np.mean(total_rewards[-100:])

            print(
                "\nlenbuffer:%d,frame:%d game:%d, reward:%.3f,mean reward: %.3f, eps:%.2f,frame/sec:%.2f"
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
                # agent._reset()
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
        if keyboard.is_pressed('p'):
            print('\npause')
            time.sleep(5)
            while True:
                if keyboard.is_pressed('p'):
                    print('reset')
                    time.sleep(1)
                    break

            # agent._reset()
        done_reward = agent.play_step(move_net, action_net, epsilon, device)
        if frame_idx % 4 == 0:
            agent.optimize_model(frame_idx, beta)

    writer.close()

    # # %%
    torch.save(move_net.state_dict(), "./checkpoints/best_move_model.pt")
    torch.save(action_net.state_dict(),
               "./checkpoints/best_action_model.pt")
    np.save("./checkpoints/frame.npy", frame_idx)
    np.save("./checkpoints/total_rewards.npy", total_rewards)
    # save buffer
    with open("./checkpoints/buffer.pickle", "wb") as f:
        pickle.dump(copy.deepcopy(buffer), f)

# %%
