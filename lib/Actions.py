# Define the actions we may need during training
# You can define your actions here

from lib.SendKey import PressKey, ReleaseKey

import time
import cv2
import threading

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
    time.sleep(0.01)


# 2
def Move_Right():
    Nothing()
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)


# 3
def Attack():
    Nothing()
    PressKey(NUM_1)
    time.sleep(0.05)
    ReleaseKey(NUM_1)
    Nothing()
    time.sleep(0.05)


# 4
def Jump():
    PressKey(UP_ARROW)
    time.sleep(0.05)
    ReleaseKey(UP_ARROW)
    Nothing()


# 5
def Roll():
    PressKey(L_SHIFT)
    time.sleep(0.05)
    ReleaseKey(L_SHIFT)


# 6
def Shield():

    Nothing()
    PressKey(NUM_2)
    time.sleep(0.05)
    ReleaseKey(NUM_2)


def Nothing_action():
    time.sleep(0.05)


# List for action functions
Actions = [Attack, Shield, Roll, Jump, Nothing_action]
Move = [Move_Left, Move_Right, Nothing]

# Run the action


def take_action(action):
    Actions[action]()


def take_move(move):
    Move[move]()
