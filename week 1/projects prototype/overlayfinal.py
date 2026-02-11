import os
import sys
import tkinter as tk
import psutil
import pynvml
import keyboard
import configparser
import time
import ctypes 

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # when running .exe
    except Exception:
        base_path = os.path.abspath(".")  # when running .py
    return os.path.join(base_path, relative_path)

APP_ID = "Overlay.Monitor.App"  # can be any unique string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

#Refresh Loop
def get_refresh_rate():
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    dc = user32.GetDC(0)
    refresh = gdi32.GetDeviceCaps(dc, 116)  # VREFRESH
    user32.ReleaseDC(0, dc)

    return refresh


#CONFIG LOAD
config=configparser.ConfigParser()
config.read(resource_path("config.ini"))

UPDATE_INTERVAL = config.getint("overlay", "update_interval_ms", fallback=500)
OPACITY = config.getfloat("overlay", "opacity", fallback=0.85)
ALWAYS_ON_TOP = config.getboolean("overlay", "always_on_top", fallback=True)
USE_BINARY = config.getboolean("units", "use_binary", fallback=True)

# -----------FPS ------------
last_time=time.perf_counter()
ui_fps=0

#Refresh Rate
display_hz=get_refresh_rate()

# ---------- GPU INIT ----------
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

# ---------- WINDOW ----------
icon_path = resource_path("overlay.ico")
root = tk.Tk()

try:
    root.iconbitmap(icon_path)
except Exception:
    pass

root.overrideredirect(True)
root.attributes("-topmost", True)
root.configure(bg="black")
root.attributes("-alpha", 0.7)     # transparency
root.geometry("+20+20")


visible = True

# ---------- LABEL ----------
label = tk.Label(
    root,
    font=("Consolas", 12),
    fg="#00ff55",
    bg="black",
    justify="left"
)
label.pack(padx=10, pady=10)

# ---------- DRAG TO MOVE ----------
x_offset = y_offset = 0

def start_move(event):
    global x_offset, y_offset
    x_offset = event.x
    y_offset = event.y

def move_window(event):
    x = root.winfo_pointerx() - x_offset
    y = root.winfo_pointery() - y_offset
    root.geometry(f"+{x}+{y}")

label.bind("<Button-1>", start_move)
label.bind("<B1-Motion>", move_window)

# ---------- HOTKEY TOGGLE ----------
def toggle_overlay():
     global visible
     if visible:
         root.withdraw() # hide 
     else: root.deiconify() # show 
     visible = not visible 


keyboard.add_hotkey("ctrl+alt+o", toggle_overlay)

# ---------- UPDATE LOOP ----------
def update():
    global last_time,ui_fps

    now=time.perf_counter()
    delta=now-last_time
    last_time=now
    if delta>0:
        ui_fps=1/delta

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
    gpu_temp = pynvml.nvmlDeviceGetTemperature(
        gpu, pynvml.NVML_TEMPERATURE_GPU
    )
    vram = pynvml.nvmlDeviceGetMemoryInfo(gpu)

    label.config(text=
        f"UI FPS:{ui_fps:5.1f}\n"
        f"DISPLAY: {display_hz} Hz\n"         
        f"CPU  : {cpu:5.1f} %\n"
        f"RAM  : {ram.used/1024**3:4.1f}/{ram.total/1024**3:.1f} GB\n"
        f"GPU  : {gpu_util:5d} %\n"
        f"VRAM : {vram.used/1024**3:4.1f}/{vram.total/1024**3:.1f} GB\n"
        f"TEMP : {gpu_temp} °C"
    )

    root.after(900, update)

update()
root.mainloop()
pynvml.nvmlShutdown()
