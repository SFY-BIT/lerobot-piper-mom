"""ATI Mini40 力传感器实时可视化"""
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

# 解析标定矩阵
tree = ET.parse(CAL_PATH)
axes = tree.findall(".//UserAxis")
M = np.array([[float(v) for v in a.get("values").split()] for a in axes])
labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]

# NI 采集任务
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5", min_val=-10.0, max_val=10.0)

# 历史缓冲区
HISTORY = 200
history = deque(maxlen=HISTORY)

def read_ft():
    raw = task.read(number_of_samples_per_channel=1)
    v = np.array([ch[0] for ch in raw])
    return M @ v

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("ATI Mini40 FT32663 — Force/Torque Sensor", fontsize=14, fontweight="bold")
bars = []
lines = []
scatters = []
texts = []

for i, (ax, label) in enumerate(zip(axes.flat, labels)):
    ax.set_title(label, fontsize=11)
    ax.set_ylim(-5, 5)  # 初始范围，会自动扩展
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 柱状图（当前值）
    bar = ax.bar(0, 0, width=0.6, color="steelblue", alpha=0.7)[0]
    bars.append(bar)
    
    # 实时曲线
    line, = ax.plot([], [], "r-", linewidth=1.5, alpha=0.8)
    lines.append(line)
    
    # 当前值文字
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                   verticalalignment="top", fontweight="bold")
    texts.append(text)
    
    # 裁剪范围指示
    ax.axhline(y=100, color="red", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.axhline(y=-100, color="red", linestyle=":", linewidth=0.5, alpha=0.3)

# 为柱状图设置单独的 x 轴范围
for ax in axes.flat:
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])

fig.tight_layout()

def update(frame):
    global history
    
    ft = read_ft()
    history.append(ft)
    
    hist_arr = np.array(list(history))
    
    for i in range(6):
        # 更新柱状图
        bars[i].set_height(ft[i])
        color = "red" if abs(ft[i]) > 100 else ("orange" if abs(ft[i]) > 50 else "steelblue")
        bars[i].set_color(color)
        
        # 更新曲线
        if len(hist_arr) > 0:
            lines[i].set_data(range(len(hist_arr)), hist_arr[:, i])
        
        # 更新文字
        texts[i].set_text(f"{ft[i]:.2f}")
        if abs(ft[i]) > 100:
            texts[i].set_color("red")
        elif abs(ft[i]) > 50:
            texts[i].set_color("orange")
        else:
            texts[i].set_color("white")
        
        # 自动调整 y 轴
        if len(hist_arr) > 0:
            ymax = max(abs(hist_arr[:, i]).max() * 1.3, 3)
            axes.flat[i].set_ylim(-ymax, ymax)
    
    return bars + lines + texts

# 键盘控制
def on_key(event):
    global history
    if event.key == "r":
        history.clear()

fig.canvas.mpl_connect("key_press_event", on_key)

print("\n按键: [R] 清除历史  [Q] 退出")
print("=" * 60)

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()

task.close()
print("采集结束")
