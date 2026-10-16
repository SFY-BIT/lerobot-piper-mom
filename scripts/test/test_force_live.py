"""连续读取力传感器，按压工具看数值是否变化"""
import numpy as np
import time
import xml.etree.ElementTree as ET
import nidaqmx

CAL_PATH = r"C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"

tree = ET.parse(CAL_PATH)
axes = tree.findall(".//UserAxis")
M = np.array([[float(v) for v in a.get("values").split()] for a in axes])

task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5", min_val=-10.0, max_val=10.0)

print("=" * 65)
print(f"{'Fx(N)':>10s} {'Fy(N)':>10s} {'Fz(N)':>10s} {'Tx(Nm)':>10s} {'Ty(Nm)':>10s} {'Tz(Nm)':>10s}")
print("-" * 65)

try:
    for i in range(200):
        raw = task.read(number_of_samples_per_channel=1)
        v = np.array([ch[0] for ch in raw])
        ft = M @ v
        print(f"{ft[0]:10.3f} {ft[1]:10.3f} {ft[2]:10.3f} {ft[3]:10.3f} {ft[4]:10.3f} {ft[5]:10.3f}")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    task.close()
    print("\n采集结束")
