from lib import Actions, GetHp
from lib.SendKey import PressKey, ReleaseKey
import time

UP_ARROW = 0x26
R = 0x52


class env:
    def __init__(self):
        self.hp_getter = GetHp.Hp_getter()
        self.win = 0

    def _reset(self):
        Actions.Nothing()
        time.sleep(10)
        print('move right')
        Actions.Move_Right()
        print('Press the R')
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        time.sleep(1)
        Actions.Nothing()

        # PressKey(UP_ARROW)
        # time.sleep(3.5)
        # ReleaseKey(UP_ARROW)
        # Actions.Move_Right()
        # time.sleep(2.5)
        # Actions.Nothing()
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        time.sleep(10)
        Actions.Move_Right()
        time.sleep(5)
        Actions.Nothing()

    def step(self, move, action, pre_player_hp, pre_Boss_hp):
        Actions.take_move(move)
        Actions.take_action(action)
        player_hp = self.hp_getter.get_self_hp()
        boss_hp = self.hp_getter.get_boss_hp()
        is_done = player_hp <= 1

        boss_damaged = 0
        player_damaged = 0

        # Win and lose conditions
        if player_hp > 1 and boss_hp <= 1:
            self.win += 1
            print(f'win,total_win:{self.win}')
            # Win
            return (100, True, player_hp, boss_hp, player_damaged, boss_damaged)

        if player_hp <= 1:
            print(f'loss,total_win:{self.win}')
            # Loss
            return (-10, True, player_hp, boss_hp, player_damaged, boss_damaged)

        boss_damaged = pre_Boss_hp - boss_hp
        player_damaged = pre_player_hp - player_hp

        # Normalize damages
        normalized_boss_damage = min(boss_damaged / 2000, 1)  # Cap at 2000
        normalized_player_damage = min(player_damaged / 1000, 1)  # Cap at 1000

        # Reward for damaging the boss
        boss_damaged_reward = 3 * normalized_boss_damage

        # Penalty for taking damage
        player_damaged_penalty = -5 * normalized_player_damage

        # Reward for dodging
        dodge_reward = 0.2 if action in {
            1, 2, 3} and player_damaged == 0 else 0

        # Calculate total reward
        total_reward = boss_damaged_reward + player_damaged_penalty + dodge_reward

        # Clip the reward to keep it within a desired range
        total_reward = max(min(total_reward, 5), -5)

        return (total_reward, is_done, player_hp, boss_hp, player_damaged, boss_damaged)
