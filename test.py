from lib import GetHelper
getter=GetHelper.getter()

while 1:
    print(abs(getter.self_pos() - getter.boss_pos()))