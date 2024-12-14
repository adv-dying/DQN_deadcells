import win32gui
import win32api
import win32process
import ctypes

Psapi = ctypes.WinDLL("Psapi.dll")
Kernel32 = ctypes.WinDLL("kernel32.dll")
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


def EnumProcessModulesEx(hProcess):
    buf_count = 256
    while True:
        LIST_MODULES_ALL = 0x03
        buf = (ctypes.wintypes.HMODULE * buf_count)()
        buf_size = ctypes.sizeof(buf)
        needed = ctypes.wintypes.DWORD()
        if not Psapi.EnumProcessModulesEx(
            hProcess,
            ctypes.byref(buf),
            buf_size,
            ctypes.byref(needed),
            LIST_MODULES_ALL,
        ):
            raise OSError("EnumProcessModulesEx failed")
        if buf_size < needed.value:
            buf_count = needed.value // (buf_size // buf_count)
            continue
        count = needed.value // (buf_size // buf_count)
        return map(ctypes.wintypes.HMODULE, buf[:count])


class Hp_getter:
    def __init__(self):
        hd = win32gui.FindWindow(None, "Dead Cells")
        pid = win32process.GetWindowThreadProcessId(hd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, pid)
        self.kernal32 = ctypes.windll.LoadLibrary(
            r"C:\\Windows\\System32\\kernel32.dll"
        )

        self.hx = 0
        # get dll address
        hProcess = Kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        hModule = EnumProcessModulesEx(hProcess)
        for i in hModule:
            temp = win32process.GetModuleFileNameEx(self.process_handle, i.value)

            if temp[-9:] == "libhl.dll":
                self.libhl = i.value

    def get_self_hp(self):
        base_address = self.libhl + 0x00048184
        offset_address = ctypes.c_long()
        offset_list = [0x4B0, 0x0, 0x58, 0x68,0x12C]
        self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            base_address,
            ctypes.byref(offset_address),
            4,
            None,
        )
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(
                int(self.process_handle),
                offset_address.value + offset,
                ctypes.byref(offset_address),
                4,
                None,
            )
        return offset_address.value

    # This function can only get hp of hornet yet
    def get_boss_hp(self):
        base_address = self.libhl + 0x00048184
        offset_address = ctypes.c_long()
        offset_list = [0x6B8,0x0, 0x18, 0x11C, 0x12C]
        self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            base_address,
            ctypes.byref(offset_address),
            4,
            None,
        )
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(
                int(self.process_handle),
                offset_address.value + offset,
                ctypes.byref(offset_address),
                4,
                None,
            )
        return offset_address.value
