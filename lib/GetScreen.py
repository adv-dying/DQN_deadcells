import torch
import cv2
import torchvision.transforms as transforms
import time
import dxcam
import win32gui
import numpy as np

class GetScreen:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.cam = dxcam.create()
        hWnd = win32gui.FindWindow(None, "Dead cells")
        left, top, right, bot = win32gui.GetWindowRect(hWnd)
        self.region=(left,top,right,bot)
        

    def grab(self):
        frames=[]
        while len(frames)<4:
            IMG= self.cam.grab(region=self.region)
            IMG=cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
            IMG=cv2.resize(IMG,(480,270))
            frames.append(torch.squeeze(self.transform(IMG)))
        
        # shape (4,270,480)  
        return torch.stack(frames).to('cuda')
    def show(self):
        IMG= self.cam.grab(region=self.region)
        IMG=cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
        IMG=cv2.resize(IMG,(480,270))
        cv2.imshow("Screenshot",IMG)
        cv2.waitKey(0)
a=GetScreen()
print(a.grab().shape)