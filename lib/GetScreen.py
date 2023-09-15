import win32gui, win32ui, win32con
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms


class GetScreen:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def grab(self):
        hWnd = win32gui.FindWindow(None, "Dead Cells")
        hWndDC = win32gui.GetWindowDC(hWnd)
        mfcDC = win32ui.CreateDCFromHandle(hWndDC)
        left, top, right, bot = win32gui.GetWindowRect(hWnd)
        width = right - left
        height = bot - top
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        im_PIL = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        im_PIL = im_PIL.resize((480,270))

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hWnd, hWndDC)
        tensor=self.transform(im_PIL)

        

        # shape (3,270,480)  
        return tensor