import win32gui
import win32api
import win32process
import ctypes
import ctypes.wintypes

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
            r"C:\Windows\System32\kernel32.dll"
        )

        # get dll address
        hProcess = Kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        hModule = EnumProcessModulesEx(hProcess)
        for i in hModule:
            temp = win32process.GetModuleFileNameEx(self.process_handle, i.value)
            if temp[-9:] == "libhl.dll":
                self.libhl = i.value

    def read(self, base, offsets):
        # Traverse pointer chain: start at base and follow each offset.
        addr = ctypes.c_long()
        self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            base,
            ctypes.byref(addr),
            4,
            None,
        )
        for offset in offsets:
            self.kernal32.ReadProcessMemory(
                int(self.process_handle),
                addr.value + offset,
                ctypes.byref(addr),
                4,
                None,
            )
        return addr

    def get_self_hp(self):
        base_address = self.libhl + 0x00048184
        offsets = [0x4B0, 0x0, 0x58, 0x68, 0x12C]
        addr = self.read(base_address, offsets)
        return addr.value

    def get_boss_hp(self):
        base_address = self.libhl + 0x00048184
        offsets = [0x6B8, 0x0, 0x18, 0x11C, 0x12C]
        addr = self.read(base_address, offsets)
        return addr.value

    def write(self, base, offsets, new_value):
        # Traverse pointer chain to get to the final address.
        addr = ctypes.c_long()
        self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            base,
            ctypes.byref(addr),
            4,
            None,
        )
        # Process all offsets except the last one.
        for offset in offsets[:-1]:
            self.kernal32.ReadProcessMemory(
                int(self.process_handle),
                addr.value + offset,
                ctypes.byref(addr),
                4,
                None,
            )
        # The final address is the current pointer plus the last offset.
        final_address = addr.value + offsets[-1]
        # Prepare the new value
        new_val = ctypes.c_long(new_value)
        bytes_written = ctypes.c_size_t(0)
        result = self.kernal32.WriteProcessMemory(
            int(self.process_handle),
            final_address,
            ctypes.byref(new_val),
            4,
            ctypes.byref(bytes_written)
        )
        if not result:
            return True
        return False

    def set_self_hp(self, new_hp):
        base_address = self.libhl + 0x00048184
        offsets = [0x4B0, 0x0, 0x58, 0x68, 0x12C]
        return self.write(base_address, offsets, new_hp)

    def set_boss_hp(self, new_hp):
        base_address = self.libhl + 0x00048184
        offsets = [0x6B8, 0x0, 0x18, 0x11C, 0x12C]
        return self.write(base_address, offsets, new_hp)

if __name__=='__main__':
    hp=Hp_getter()
    print(hp.get_self_hp())
    hp.set_self_hp(20000)
    print(hp.get_boss_hp())
    hp.set_boss_hp(20000)