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
MAX_EPISODE = int(input('play_turn:'))
ESC = 0x1B

epsilon_final = 0.02
epsilon_decay_frame = 2 * 10**3
replay_min_size = 200
batch_size = 32
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
        self.total_rewards = 0.0
        self.bosshp = self.hpgetter.get_boss_hp()
        self.playerhp = self.hpgetter.get_self_hp()
        self.shadow = np.stack((state, state, state, state), axis=0)

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
        new_state, reward, is_done, self.playerhp, self.bosshp = self.env.step(
            action, self.playerhp, self.bosshp
        )
        print(action)
        print(reward)
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
            time.sleep(11)

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


net = DQN(7).to(device)
tgt_net = DQN(7).to(device)
# load the model
try:
    net.load_state_dict(torch.load("./checkpoints/best_model.pt"))
    frame_idx = np.load("./checkpoints/frame.npy")
    tgt_net.load_state_dict(net.state_dict())
    print(frame_idx)
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
    buffer = ExperienceBuffer(capacity=400)
    print("new buffer")
agent = Agent(buffer)
total_rewards = []
best_mean_reward = None

reward = None
# numbers of game
time_start=time.time()
while len(total_rewards) < MAX_EPISODE:
    frame_idx += 1
    epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_frame)

    if reward is not None:
        # one loop ends
        # if something goes wrong, break
        if reward==0.0:
            break
        # optimize
        if len(buffer) > replay_min_size and len(total_rewards) % 5 == 4:
            optimizer.zero_grad()
            batch = buffer.sample(batch_size)
            loss_t = agent.cal_loss(batch, net, tgt_net)
            loss_t.backward()
            optimizer.step()
            time.sleep(10)

        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-100:])
        print(
            "%d:done %d:game, mean reward: %.3f, eps:%.2f"
            % (frame_idx, len(total_rewards), mean_reward, epsilon)
        )
        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        # save model
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), "./checkpoints/best_model.pt")

            if best_mean_reward is not None:
                print(
                    "Best mean reward updated %.3f -> %.3f, model saved"
                    % (best_mean_reward, mean_reward)
                )
            best_mean_reward = mean_reward
        # reset game

        agent._reset()

        np.save("./checkpoints/frame.npy", frame_idx)
        if len(total_rewards) % 10 == 0:
            tgt_net.load_state_dict(net.state_dict())

        # save buffer
        with open("./checkpoints/buffer.pickle", "wb") as f:
            pickle.dump(copy.deepcopy(buffer), f)

    # step between copy the net to tgt_net
    reward = agent.play_step(net, epsilon, device)


writer.close()
print(time.localtime())
print((time.time()-time_start)/60)
