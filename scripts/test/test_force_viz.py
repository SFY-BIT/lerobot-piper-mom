"""ATI Mini40 力传感器实时可视化（支持全局归零）"""
import os
import numpy as np
import time
import xml.etree.ElementTree as ET
from collections import deque
import nidaqmx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CAL_PATH = r"C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"
BASE_PATH = os.path.join(os.path.dirname(CAL_PATH), "FT32663_tare.npy")
TOOL_PATH = os.path.join(os.path.dirname(CAL_PATH), "FT32663_tool.npy")

# 解析标定矩阵
tree = ET.parse(CAL_PATH)
axes = tree.findall(".//UserAxis")
M = np.array([[float(v) for v in a.get("values").split()] for a in axes])
labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]

# === 双段归零 ===
base_tare = np.load(BASE_PATH) if os.path.exists(BASE_PATH) else np.zeros(6)
tool_tare = np.load(TOOL_PATH) if os.path.exists(TOOL_PATH) else np.zeros(6)
print(f"基础偏置: {np.round(base_tare, 2)}")
print(f"工具偏置: {np.round(tool_tare, 2)}")

# NI 采集任务
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5", min_val=-10.0, max_val=10.0)

HISTORY = 200
history = deque(maxlen=HISTORY)

def read_raw():
    raw = task.read(number_of_samples_per_channel=1)
    v = np.array([ch[0] for ch in raw])
    return M @ v

def read_ft():
    return read_raw() - base_tare - tool_tare

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("ATI Mini40 FT32663 (按 T 归零并保存到文件)", fontsize=12, fontweight="bold")
bars = []
lines = []
texts_t = []
texts_r = []

for i, (ax, label) in enumerate(zip(axes.flat, labels)):
    ax.set_title(label, fontsize=11)
    ax.set_ylim(-3, 3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    bar = ax.bar(0, 0, width=0.6, color="steelblue", alpha=0.7)[0]
    bars.append(bar)
    line, = ax.plot([], [], "r-", linewidth=1.5, alpha=0.8)
    lines.append(line)
    txt1 = ax.text(0.02, 0.88, "", transform=ax.transAxes, fontsize=11,
                   verticalalignment="top", fontweight="bold", color="lime")
    texts_t.append(txt1)
    txt2 = ax.text(0.02, 0.05, "", transform=ax.transAxes, fontsize=8,
                   verticalalignment="bottom", color="gray")
    texts_r.append(txt2)

for ax in axes.flat:
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])

fig.tight_layout()
fig.subplots_adjust(top=0.92)

def update(frame):
    raw = read_raw()
    ft = read_ft()
    history.append(ft)
    hist_arr = np.array(list(history))
    for i in range(6):
        bars[i].set_height(ft[i])
        absv = abs(ft[i])
        bars[i].set_color("red" if absv > 50 else ("orange" if absv > 20 else "steelblue"))
        if len(hist_arr) > 0:
            lines[i].set_data(range(len(hist_arr)), hist_arr[:, i])
        texts_t[i].set_text(f"{ft[i]:.2f}")
        texts_t[i].set_color("red" if absv > 50 else ("orange" if absv > 20 else "lime"))
        texts_r[i].set_text(f"raw: {raw[i]:.1f}")
        if len(hist_arr) > 0:
            ymax = max(abs(hist_arr[:, i]).max() * 1.3, 3)
            axes.flat[i].set_ylim(-ymax, ymax)
    return bars + lines + texts_t + texts_r

def on_key(event):
    global base_tare, tool_tare, history
    if event.key == "b":
        print("基础归零（无工具，100帧平均）...")
        vals = [read_raw() for _ in range(100)]
        base_tare = np.mean(vals, axis=0)
        np.save(BASE_PATH, base_tare)
        tool_tare = np.zeros(6)
        if os.path.exists(TOOL_PATH):
            os.remove(TOOL_PATH)
        history.clear()
        print(f"基础偏置已保存: {np.round(base_tare, 2)}")
    elif event.key == "t":
        print("工具归零（装工具后，100帧平均）...")
        vals = [read_raw() for _ in range(100)]
        raw_mean = np.mean(vals, axis=0)
        tool_tare = raw_mean - base_tare
        np.save(TOOL_PATH, tool_tare)
        history.clear()
        print(f"工具偏置已保存: {np.round(tool_tare, 2)}")
    elif event.key == "r":
        history.clear()
    elif event.key == "z":
        tool_tare = np.zeros(6)
        if os.path.exists(TOOL_PATH):
            os.remove(TOOL_PATH)
        history.clear()
        print("工具偏置已清除")

fig.canvas.mpl_connect("key_press_event", on_key)

print("=" * 60)
print("双段归零: [B] 基础(无工具)  [T] 工具(装夹爪后)  [Z] 清除工具  [Q] 退出")
print("绿色=接触力  灰色=原始值(含偏置)")
print("=" * 60)

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()

task.close()
print("结束")
