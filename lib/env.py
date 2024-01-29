from lib import GetScreen, Actions, GetHelper
from lib.SendKey import PressKey, ReleaseKey
import threading
import time
import math

UP_ARROW = 0x26
RIGHT_ARROW = 0x27
R = 0x52


def hit(pre_hp, hp):
    if pre_hp > hp:
        return True
    else:
        return False
    
def _Move_Right():
    Actions.Nothing()
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)

class env:

    def __init__(self):
        self.screen = GetScreen.GetScreen()

        self.getter = GetHelper.getter()

    def _reset(self):
        Actions.Nothing()
        time.sleep(4)
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        _Move_Right()
        time.sleep(1.3)
        # Actions.Nothing()
        # PressKey(UP_ARROW)
        # time.sleep(3.5)
        # ReleaseKey(UP_ARROW)
        # _Move_Right()
        # time.sleep(2.5)
        # Actions.Nothing()
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        Actions.Nothing()
        time.sleep(8)
        _Move_Right()
        time.sleep(6)
        Actions.Nothing()
        return self.screen.grab()

    def step(
        self,
        attack,
        move,
        pre_player_hp,
        pre_Boss_hp,
    ):
        Actions.take_actions(attack, move)

        time.sleep(0.1)

        player_hp = self.getter.self_hp()
        boss_hp = self.getter.boss_hp()

        hitted = hit(pre_Boss_hp, boss_hp)
        be_hitted = hit(pre_player_hp, player_hp)

        move_reward = 0
        hit_reward = 0

        if be_hitted:
            move_reward -= 10
            hit_reward -= 10
        distance = abs(self.getter.self_pos() - self.getter.boss_pos())

        if distance > 5:
            move_reward -= 1
        elif distance <= 2:
            move_reward -= 2
        else:
            move_reward += 2

        if boss_hp <= 1:
            if player_hp > 1:
                print(player_hp)
                move_reward += 200
                hit_reward += 200
                print("pass")
            else:
                move_reward -= 200
                hit_reward -= 200
                print("fail")

        if attack == 0:
            if abs(distance > 3):
                hit_reward -= 3
            if hitted:
                hit_reward += 4

        elif attack == 1 or attack == 2:
            if abs(distance > 4):
                hit_reward -= 2
            if hitted:
                hit_reward += 7
            else:
                hit_reward -= 2

        return (
            self.screen.grab(),
            move_reward,
            hit_reward,
            boss_hp <= 1,
            player_hp,
            boss_hp,
        )
