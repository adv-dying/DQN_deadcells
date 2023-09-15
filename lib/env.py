from lib import GetScreen, Actions, GetHp
from lib.SendKey import PressKey, ReleaseKey
import threading
import time

UP_ARROW = 0x26
R = 0x52


class env:
    def __init__(self):
        self.screen = GetScreen.GetScreen()
        self.hp_getter = GetHp.Hp_getter()

    def _reset(self):
        Actions.Nothing()
        time.sleep(4)
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        Actions.Move_Right()
        time.sleep(0.4)
        Actions.Nothing()
        PressKey(UP_ARROW)
        time.sleep(3.5)
        ReleaseKey(UP_ARROW)
        Actions.Move_Right()
        time.sleep(2.5)
        Actions.Nothing()
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        time.sleep(10)
        Actions.Move_Right()
        time.sleep(8)
        Actions.Nothing()
        return self.screen.grab()

    def step(self, action, pre_player_hp, pre_Boss_hp):
        Actions.take_action(action)
        time.sleep(0.1)
        player_hp = self.hp_getter.get_self_hp()
        boss_hp = self.hp_getter.get_boss_hp()
        if boss_hp <= 1:
            return (self.screen.grab(), 0, boss_hp <= 1, player_hp, boss_hp)
        # action 0:Shield
        if action == 0:
            if (pre_Boss_hp - boss_hp) > 0:
                return (
                    self.screen.grab(),
                    (pre_Boss_hp - boss_hp) * 0.003
                    - (pre_player_hp - player_hp) * 0.01,
                    boss_hp <= 1,
                    player_hp,
                    boss_hp,
                )
            else:
                return (
                    self.screen.grab(),
                    -4 - (pre_player_hp - player_hp) * 0.01,
                    boss_hp <= 1,
                    player_hp,
                    boss_hp,
                )
        # return state,reward,is_done
        # action 1: Roll
        elif action == 1:
            return (
                self.screen.grab(),
                -(pre_player_hp - player_hp) * 0.012,
                boss_hp <= 1,
                player_hp,
                boss_hp,
            )
        if action == 2:
            if (pre_Boss_hp - boss_hp) > 0:
                return (
                    self.screen.grab(),
                    (
                        (pre_Boss_hp - boss_hp) * 0.002
                        - (pre_player_hp - player_hp) * 0.02
                    ),
                    boss_hp <= 1,
                    player_hp,
                    boss_hp,
                )
            else:
                return (
                    self.screen.grab(),
                    (-5 - (pre_player_hp - player_hp) * 0.02),
                    boss_hp <= 1,
                    player_hp,
                    boss_hp,
                )
        return (
            self.screen.grab(),
            -(pre_player_hp - player_hp) * 0.01,
            boss_hp <= 1,
            player_hp,
            boss_hp,
        )
