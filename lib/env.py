from lib import Actions, GetHp
from lib.SendKey import PressKey, ReleaseKey
import time

UP_ARROW = 0x26
R = 0x52


class env:
    def __init__(self):
        self.hp_getter = GetHp.Hp_getter()

    def _reset(self):
        Actions.Nothing()
        time.sleep(4)
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        Actions.Move_Right()
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
        

    def step(self, action, pre_player_hp, pre_Boss_hp):
        Actions.take_action(action)
        player_hp = self.hp_getter.get_self_hp()
        boss_hp = self.hp_getter.get_boss_hp()
        if boss_hp <= 1 and player_hp>1:
            return (100, True, player_hp, boss_hp)
        if boss_hp <= 1 and player_hp<=1 :
            return (-100, True, player_hp, boss_hp)
        return ((pre_Boss_hp-boss_hp)*0.0007192-(pre_player_hp-player_hp)*0.01, player_hp <= 1, player_hp, boss_hp)
        # self.hp=15482
        # boss.hp=215249
        # 15482/215249=0.07192
        # new_state, reward, is_done, self.playerhp, self.bosshp
