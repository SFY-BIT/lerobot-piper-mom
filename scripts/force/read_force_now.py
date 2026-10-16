"""读取当前力传感器实时值 (标定后 = 接触力)"""
import numpy as np
import xml.etree.ElementTree as ET
import nidaqmx
import time
import os

CAL = r"C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"
dirp = os.path.dirname(CAL)
tree = ET.parse(CAL)
axes = tree.findall(".//UserAxis")
M = np.array([[float(v) for v in a.get("values").split()] for a in axes])
base = np.load(os.path.join(dirp, "FT32663_tare.npy"))
tool = np.load(os.path.join(dirp, "FT32663_tool.npy"))
print(f"当前偏置: base={np.round(base,2)} tool={np.round(tool,2)}")
print()

task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5", min_val=-10.0, max_val=10.0)
print("=== 当前空闲读数 (标定后 = 接触力, 悬空应≈0) ===")
vals = []
for i in range(15):
    raw = task.read(number_of_samples_per_channel=1)
    v = np.array([ch[0] for ch in raw])
    ft = M @ v - base - tool
    vals.append(ft)
    print(f"  Fx={ft[0]:+6.2f} Fy={ft[1]:+6.2f} Fz={ft[2]:+6.2f} Tx={ft[3]:+.2f} Ty={ft[4]:+.2f} Tz={ft[5]:+.2f}")
    time.sleep(0.1)
task.close()
v = np.array(vals)
print(f"\n均值: {np.round(v.mean(axis=0), 2)}")
print(f"std:  {np.round(v.std(axis=0), 3)}")
