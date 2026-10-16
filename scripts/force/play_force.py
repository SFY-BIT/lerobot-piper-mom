"""力+双视角视频播放器 — 左视频1 中视频2 右6轴柱状图

用法:
  python play_force.py
  python play_force.py my_robot/stamp_seal_v2

键盘:
  SPACE   暂停/播放
  ← →    前后跳转1秒
  ↑ ↓    调速
  [ ]    前后切换 episode
  R       重新播放当前
  ESC     退出
"""

import sys
from pathlib import Path
import json
import numpy as np
import cv2
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import datetime

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_DATASET = "my_robot/stamp_seal_v2"

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
})


def load_episode_data(dataset_path: Path, episode: int):
    """加载单个episode的数据和视频"""
    pq_path = dataset_path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not pq_path.exists():
        return None

    t = pq.read_table(str(pq_path))
    ts = np.array(t["timestamp"])
    force = np.stack([np.array(x) for x in t["observation.force"].to_pylist()])
    n = len(ts)

    # 找两个摄像头的视频
    video_base = dataset_path / "videos" / "chunk-000"
    imgs_cam1, imgs_cam2 = {}, {}
    for cam_name, imgs_dict in [("observation.images.one", imgs_cam1),
                                 ("observation.images.two", imgs_cam2)]:
        matches = list(video_base.glob(f"{cam_name}/episode_{episode:06d}.mp4"))
        if matches:
            cap = cv2.VideoCapture(str(matches[0]))
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, fr = cap.read()
                if ret:
                    imgs_dict[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cap.release()

    return {
        "ts": ts - ts[0],
        "force": force,
        "n": n,
        "imgs1": imgs_cam1,
        "imgs2": imgs_cam2,
    }


def get_episode_list(dataset_path: Path) -> list[int]:
    """获取数据集中所有episode编号"""
    data_dir = dataset_path / "data" / "chunk-000"
    eps = sorted([int(f.stem.split("_")[1]) for f in data_dir.glob("episode_*.parquet")])
    return eps


def main():
    dataset_name = DEFAULT_DATASET
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            dataset_name = a; break

    dataset_path = CACHE / dataset_name
    episodes = get_episode_list(dataset_path)
    if not episodes:
        print(f"No episodes found in {dataset_name}")
        return

    cur_ep = episodes[0]
    data = load_episode_data(dataset_path, cur_ep)
    if data is None:
        return

    print(f"Dataset: {dataset_name}  Episodes: {episodes[0]}-{episodes[-1]} ({len(episodes)} total)")
    print(f"SPACE=pause  ←→=skip  ↑↓=speed  []=change ep  R=replay  ESC=quit")

    labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    # ── 布局 ──
    fig = plt.figure(figsize=(20, 8))
    fig.canvas.manager.set_window_title(f"Force Player — {dataset_name}")

    # 左视频 cam1
    ax1 = fig.add_axes([0.01, 0.10, 0.32, 0.87])
    ax1.set_facecolor("#000"); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Camera 1", color="#58a6ff", fontsize=10, pad=2)
    vid1 = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    info1 = ax1.text(0.01, 0.99, "", transform=ax1.transAxes, ha="left", va="top",
                     fontsize=9, fontfamily="monospace", color="#58a6ff",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.8,
                               edgecolor="#58a6ff", linewidth=1.5))

    # 中视频 cam2
    ax2 = fig.add_axes([0.34, 0.10, 0.32, 0.87])
    ax2.set_facecolor("#000"); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Camera 2", color="#3fb950", fontsize=10, pad=2)
    vid2 = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    info2 = ax2.text(0.01, 0.99, "", transform=ax2.transAxes, ha="left", va="top",
                     fontsize=9, fontfamily="monospace", color="#3fb950",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.8,
                               edgecolor="#3fb950", linewidth=1.5))

    # 右6轴柱状图
    bar_axes, bar_bars, bar_txts = [], [], []
    for i in range(6):
        ax = fig.add_axes([0.69 + (i % 3) * 0.10, 0.55 - (i // 3) * 0.42, 0.09, 0.38])
        ax.set_facecolor("#161b22")
        ax.set_title(labels[i], color="white", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.grid(True, alpha=0.1, axis="y", color="white")
        ax.axhline(y=0, color="white", alpha=0.3, linewidth=1)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        bar = ax.bar(0, 0, width=0.7, color=colors[i], edgecolor="white", linewidth=0.4)[0]
        txt = ax.text(0, 0, "", ha="center", fontsize=8, fontweight="bold", color="white")
        bar_axes.append(ax); bar_bars.append(bar); bar_txts.append(txt)

    status = fig.text(0.5, 0.01, "", ha="center", fontsize=8, color="#8b949e", fontfamily="monospace")

    # ── 状态（用闭包捕获可变引用） ──
    state = {"idx": 0, "paused": False, "speed": 1.0, "t_start": 0.0}
    ep_ref = [cur_ep]  # 可变容器

    # ── 切换 episode ──
    def switch_ep(new_ep):
        new_data = load_episode_data(dataset_path, new_ep)
        if new_data is None:
            return False
        nonlocal data
        data = new_data
        ep_ref[0] = new_ep
        state["idx"] = 0
        state["t_start"] = datetime.datetime.now().timestamp()
        state["paused"] = False
        return True

    def update():
        d = data; s = state
        n = d["n"]; ts = d["ts"]; force = d["force"]

        if not s["paused"]:
            elapsed = (datetime.datetime.now().timestamp() - s["t_start"]) * s["speed"]
            s["idx"] = int(elapsed * 30)
            if s["idx"] >= n:
                s["idx"] = 0; s["t_start"] = datetime.datetime.now().timestamp()  # 循环

        i = s["idx"]; ft = force[i]; ct = ts[i]

        # 视图标注
        if ft[2] < -10:    ph, phc = "HEAVY", "#f85149"
        elif ft[2] < -3:   ph, phc = "PRESS", "#d29922"
        elif ft[2] > 3:    ph, phc = "LIFT", "#3fb950"
        else:              ph, phc = "IDLE", "#8b949e"

        info_txt = f"F{i}/{n-1}  t={ct:.1f}s\nFz={ft[2]:+.1f}N [{ph}]"
        info1.set_text(info_txt); info1.set_color(phc)
        info2.set_text(info_txt); info2.set_color(phc)

        # 两路视频
        if i in d["imgs1"]: vid1.set_data(d["imgs1"][i])
        if i in d["imgs2"]: vid2.set_data(d["imgs2"][i])

        # 柱状图
        for j in range(6):
            v = ft[j]; lim = 30 if j < 3 else 1.5
            mx = max(abs(v) * 1.4, lim * 0.3)
            bar_axes[j].set_ylim(-mx, mx)
            bar_bars[j].set_height(v); bar_bars[j].set_y(min(0, v))
            fmt = f"{v:.3f}" if j >= 3 else f"{v:.1f}"
            bar_txts[j].set_y(v + mx * 0.06 if v >= 0 else v - mx * 0.06)
            bar_txts[j].set_text(fmt)
            bar_txts[j].set_va("bottom" if v >= 0 else "top")

        ps = "PAUSED" if s["paused"] else f"{s['speed']:.1f}x"
        status.set_text(
            f"Ep{ep_ref[0]}  F{i}/{n-1}  t={ct:.1f}s  "
            f"Fz={ft[2]:+.1f}  Fx={ft[0]:+.1f}  Fy={ft[1]:+.1f}  "
            f"Tx={ft[3]:+.3f}  Ty={ft[4]:+.3f}  Tz={ft[5]:+.3f}  {ps}"
        )

        fig.canvas.draw_idle()

    def on_key(e):
        if e.key is None: return
        s = state; d = data
        n = d["n"]

        if e.key == "right":
            s["idx"] = min(n-1, s["idx"]+30); s["t_start"] = datetime.datetime.now().timestamp()-s["idx"]/30.0
        elif e.key == "left":
            s["idx"] = max(0, s["idx"]-30); s["t_start"] = datetime.datetime.now().timestamp()-s["idx"]/30.0
        elif e.key in ("shift+right", "ctrl+right"):
            s["idx"] = min(n-1, s["idx"]+5); s["t_start"] = datetime.datetime.now().timestamp()-s["idx"]/30.0
        elif e.key in ("shift+left", "ctrl+left"):
            s["idx"] = max(0, s["idx"]-5); s["t_start"] = datetime.datetime.now().timestamp()-s["idx"]/30.0
        elif e.key == " ":
            s["paused"] = not s["paused"]
            if not s["paused"]:
                s["t_start"] = datetime.datetime.now().timestamp()-s["idx"]/30.0/s["speed"]
        elif e.key == "up":
            s["speed"] = min(4.0, s["speed"]+0.25)
        elif e.key == "down":
            s["speed"] = max(0.25, s["speed"]-0.25)
        elif e.key in ("[", "pageup"):
            # 上一ep
            idx = episodes.index(ep_ref[0])
            if idx > 0:
                switch_ep(episodes[idx-1])
                print(f"  -> Episode {ep_ref[0]}")
        elif e.key in ("]", "pagedown"):
            # 下一ep
            idx = episodes.index(ep_ref[0])
            if idx < len(episodes)-1:
                switch_ep(episodes[idx+1])
                print(f"  -> Episode {ep_ref[0]}")
        elif e.key == "r":
            switch_ep(ep_ref[0])  # 重新加载当前
            print(f"  Replay Episode {ep_ref[0]}")
        elif e.key == "home":
            s["idx"] = 0; s["t_start"] = datetime.datetime.now().timestamp()
        elif e.key == "end":
            s["idx"] = n-1; s["paused"] = True
        elif e.key == "escape":
            plt.close("all"); return
        update()
        # 更新窗口标题
        fig.canvas.manager.set_window_title(
            f"Force Player — {dataset_name}  Ep{ep_ref[0]}"
        )

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=33)
    timer.add_callback(update)
    timer.start()

    state["t_start"] = datetime.datetime.now().timestamp()
    update()
    plt.show()


if __name__ == "__main__":
    main()
