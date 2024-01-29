# Define the actions we may need during training
# You can define your actions here

from lib.SendKey import PressKey, ReleaseKey
from lib.WindowsAPI import grab_screen
import time
import cv2
from threading import Thread

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LEFT_ARROW = 0x25
RIGHT_ARROW = 0x27

SPACE = 0x20
L_SHIFT = 0xA0
NUM_1 = 0x31
NUM_2 = 0x32


# 0
def Nothing():
    ReleaseKey(LEFT_ARROW)
    ReleaseKey(RIGHT_ARROW)
    pass


# 1
def Move_Left():
    Nothing()
    PressKey(LEFT_ARROW)
    time.sleep(0.2)
    Nothing()


# 2
def Move_Right():
    Nothing()
    PressKey(RIGHT_ARROW)
    time.sleep(0.2)
    Nothing()


# 3
def Attack():
    Nothing()
    PressKey(NUM_1)
    time.sleep(0.1)
    ReleaseKey(NUM_1)
    Nothing()



# 4
def Shield_left():
    Nothing()
    Move_Left()
    Nothing()
    PressKey(NUM_2)
    time.sleep(0.1)
    ReleaseKey(NUM_2)
    time.sleep(0.1)


# 5
def Shield_right():
    Nothing()
    Move_Right()
    Nothing()
    PressKey(NUM_2)
    time.sleep(0.1)
    ReleaseKey(NUM_2)
    time.sleep(0.1)


# 6
def Jump():
    PressKey(UP_ARROW)
    time.sleep(0.1)
    ReleaseKey(UP_ARROW)
    Nothing()


# 7
def Roll():
    PressKey(L_SHIFT)
    time.sleep(0.5)
    ReleaseKey(L_SHIFT)


# List for action functions
attacks = [
    Attack,
    Shield_left,
    Shield_right,
    Nothing,
]

moves = [
    Nothing,
    Move_Left,
    Move_Right,
    Jump,
    Roll,
]


def attack_actions(action):
    attacks[action]()


def move_actions(action):
    moves[action]()


def take_actions(attack_action, move_action):
    # t1 = Thread(target=attack_actions, args=(attack_action, ))
    # t2 = Thread(target=move_actions, args=(move_action, ))

    # t2.start()
    # t1.start()


    # t1.join()
    # t2.join()
    move_actions(move_action)
    attack_actions(attack_action)
