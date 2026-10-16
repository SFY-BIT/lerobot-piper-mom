"""Ep0 撞击事件 — 力+关节运动 交互式逐帧浏览器（宽范围版）

布局:
  左上: 大屏视频帧 + 力值叠加
  右上: 6轴力柱状图
  左中: 关节位置变化曲线 (J2肩, J3肘 — 垂直运动相关)
  右中: 关节速度 (说明运动与力的因果关系)
  左下: Fz 全景 + 光标
  右下: Fz 撞击区放大 + 光标

键盘: LEFT/RIGHT=±5帧  SHIFT+LEFT/RIGHT=±1帧  SPACE=播放
       UP/DOWN=调速  HOME/END=跳转  ESC=退出
"""

import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))
from piper_fk import forward_kinematics, compute_eef_trajectory

PARQUET = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_v2\data\chunk-000\episode_000000.parquet"
VIDEO_BASE = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_v2\videos\chunk-000"

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
})


def main():
    # ── 加载 ──
    t = pq.read_table(PARQUET)
    ts = np.array(t["timestamp"])
    force = np.stack([np.array(x) for x in t["observation.force"].to_pylist()])
    state = np.stack([np.array(x) for x in t["observation.state"].to_pylist()])
    t_rel = ts - ts[0]
    fz = force[:, 2]
    n_total = len(ts)

    # 正运动学计算EEF位置
    joints = state[:, :6]
    eef_xyz, eef_vel = compute_eef_trajectory(joints)
    eef_z = eef_xyz[:, 2]  # Z轴（垂直）最相关
    eef_speed = eef_vel

    # ── 视频 ──
    video_matches = list(Path(VIDEO_BASE).glob("**/episode_000000.mp4"))
    cap = cv2.VideoCapture(str(video_matches[0]))

    # ── 范围（宽范围 200帧） ──
    lo, hi = 200, 400
    n_seq = hi - lo + 1
    print(f"Browser: frames {lo}->{hi} ({n_seq} frames, ~{n_seq/30:.1f}s)")
    print(f"Impact zone: Fz min={fz[lo:hi+1].min():.1f}N @ F{lo + np.argmin(fz[lo:hi+1])}")
    print(f"  LEFT/RIGHT=5f  SHIFT+LR=1f  SPACE=play  ESC=quit")

    # ── 预加载视频 ──
    imgs = {}
    for frm in range(lo, hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
        ret, img = cap.read()
        if ret:
            imgs[frm] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()
    print(f"Preloaded {len(imgs)}/{n_seq} video frames")

    # ── 状态 ──
    state_dict = {"idx": 0, "paused": True, "speed": 1.0, "t_play_start": 0.0}

    force_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    force_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    joint_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#888"]
    joint_names = ["J1(Base)", "J2(Shldr)", "J3(Elbow)", "J4(Wrist1)", "J5(Wrist2)", "J6(Wrist3)", "J7(Grip)"]

    # ── 布局 3行×2列 ──
    fig = plt.figure(figsize=(22, 13))
    fig.canvas.manager.set_window_title("Ep0 Impact — Force + Joint Motion Browser")
    fig.suptitle("EPISODE 0 (v2) — Gripper Smashes Table  |  Force + EEF Motion Alignment (Forward Kinematics)",
                 fontsize=14, fontweight="bold", color="white", y=0.995)

    gs = plt.GridSpec(3, 2, figure=fig, width_ratios=[2.5, 1.5],
                      height_ratios=[2.5, 1.8, 1.5], hspace=0.35, wspace=0.2)

    # (0,0) 大屏视频
    ax_vid = fig.add_subplot(gs[0, 0])
    ax_vid.set_facecolor("#000")
    ax_vid.set_xticks([]); ax_vid.set_yticks([])
    vid_img = ax_vid.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    vid_text = ax_vid.text(0.01, 0.99, "", transform=ax_vid.transAxes,
                           ha="left", va="top", fontsize=11, fontfamily="monospace",
                           color="#58a6ff", fontweight="bold",
                           bbox=dict(boxstyle="round", facecolor="black", alpha=0.8,
                                     edgecolor="#58a6ff", linewidth=2))

    # (0,1) 6轴力 + 关节速度
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor("#161b22")
    bars = ax_bar.bar(force_labels, np.zeros(6), color=force_colors,
                      edgecolor="white", linewidth=0.8, width=0.6)
    ax_bar.axhline(y=0, color="white", alpha=0.3, linewidth=1)
    ax_bar.set_title("6-Axis Force (current frame)", fontsize=11, color="white", fontweight="bold")
    ax_bar.grid(True, alpha=0.1, axis="y", color="white")
    bar_lbl = [ax_bar.text(i, 0, "", ha="center", fontsize=9, fontweight="bold", color="white")
               for i in range(6)]

    # (1,0) EEF位置 (Z轴高度 + Fz 双轴) — 正运动学计算
    ax_eef = fig.add_subplot(gs[1, 0])
    ax_eef.set_facecolor("#161b22")
    eef_z_line, = ax_eef.plot(t_rel, eef_z, color="#f39c12", linewidth=1.2, alpha=0.7, label="EEF Z (m)")
    ax_eef.set_ylabel("EEF Height Z (m)", fontsize=9, color="#f39c12")
    ax_eef.tick_params(axis="y", colors="#f39c12")
    ax_eef.set_title("EEF Position Z (Forward Kinematics)  vs  Fz — when arm drops, force spikes",
                     fontsize=11, color="white")
    ax_eef.grid(True, alpha=0.1, color="white")
    ax_eef.set_xlim(t_rel[0], t_rel[-1])

    # Fz 双轴
    ax_fz_twin = ax_eef.twinx()
    fz_line, = ax_fz_twin.plot(t_rel, fz, color="#3498db", linewidth=1.2, alpha=0.5, label="Fz (N)")
    ax_fz_twin.set_ylabel("Fz (N)", fontsize=9, color="#3498db")
    ax_fz_twin.tick_params(axis="y", colors="#3498db")

    ax_eef.legend([eef_z_line, fz_line], ["EEF Z", "Fz"], fontsize=8, loc="upper right",
                  facecolor="#161b22", edgecolor="#333", labelcolor="white")

    cursor_eef = ax_eef.axvline(x=t_rel[lo], color="#f85149", linewidth=2, alpha=0.7)
    cursor_fz1 = ax_fz_twin.axvline(x=t_rel[lo], color="#f85149", linewidth=2, alpha=0.3)

    # (1,1) EEF速度 + Fz变化率 — 因果性证明
    ax_eefv = fig.add_subplot(gs[1, 1])
    ax_eefv.set_facecolor("#161b22")
    ax_eefv.plot(t_rel, eef_speed, color="#f39c12", linewidth=1.2, alpha=0.6, label="EEF Speed (m/s)")
    ax_eefv.set_ylabel("EEF Speed (m/s)", fontsize=9, color="#f39c12")
    ax_eefv.tick_params(axis="y", colors="#f39c12")
    ax_eefv.set_title("EEF Speed — movement peaks align with force changes", fontsize=11, color="white")
    ax_eefv.set_xlabel("Time (s)", fontsize=9, color="white")

    # |dFz/dt| 双轴
    ax_dfz_twin = ax_eefv.twinx()
    dfz = np.abs(np.diff(fz, append=fz[-1])) / 0.03333
    ax_dfz_twin.plot(t_rel, dfz, color="#3498db", linewidth=0.8, alpha=0.4, label="|dFz/dt| (N/s)")
    ax_dfz_twin.set_ylabel("|dFz/dt| (N/s)", fontsize=9, color="#3498db")
    ax_dfz_twin.tick_params(axis="y", colors="#3498db")

    ax_eefv.legend(fontsize=7, facecolor="#161b22", edgecolor="#333", labelcolor="white")
    ax_eefv.grid(True, alpha=0.1, color="white")
    ax_eefv.set_xlim(t_rel[0], t_rel[-1])
    cursor_eefv = ax_eefv.axvline(x=t_rel[lo], color="#f85149", linewidth=2, alpha=0.7)

    # (2,0) Fz 全景
    ax_fz = fig.add_subplot(gs[2, 0])
    ax_fz.plot(t_rel, fz, color="#3498db", linewidth=1.0, alpha=0.6)
    ax_fz.axvspan(t_rel[lo], t_rel[hi], alpha=0.06, color="#f39c12")
    ax_fz.set_title("Fz — Full Episode  (orange = browsing range)", fontsize=11, color="white")
    ax_fz.set_xlabel("Time (s)", fontsize=9, color="white")
    ax_fz.set_ylabel("Fz (N)", fontsize=9, color="white")
    ax_fz.grid(True, alpha=0.1, color="white")
    ax_fz.set_xlim(t_rel[0], t_rel[-1])
    cursor_fz = ax_fz.axvline(x=t_rel[lo], color="#f85149", linewidth=2.5, alpha=0.8)
    dot_fz = ax_fz.plot(t_rel[lo], fz[lo], "o", color="#f85149", markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5)[0]

    # (2,1) 局部放大 200-400
    ax_zoom = fig.add_subplot(gs[2, 1])
    tz = t_rel[lo:hi + 1]
    ax_zoom.plot(tz, fz[lo:hi + 1], color="#3498db", linewidth=1.3, marker="o",
                 markersize=2, markerfacecolor="white", alpha=0.5)
    ax_zoom.set_title("Fz — Browsing Zone (F200-400)", fontsize=11, color="white")
    ax_zoom.set_xlabel("Time (s)", fontsize=9, color="white")
    ax_zoom.grid(True, alpha=0.1, color="white")
    cursor_zoom = ax_zoom.axvline(x=t_rel[lo], color="#f85149", linewidth=2.5, alpha=0.8)
    dot_zoom = ax_zoom.plot(t_rel[lo], fz[lo], "o", color="#f85149", markersize=12,
                            markeredgecolor="white", markeredgewidth=2)[0]

    # 状态栏
    status_text = fig.text(0.5, 0.001, "", ha="center", fontsize=8, color="#8b949e",
                           fontfamily="monospace")

    # ── 更新 ──
    def update():
        s = state_dict
        if not s["paused"]:
            elapsed = (datetime.datetime.now().timestamp() - s["t_play_start"]) * s["speed"]
            s["idx"] = min(n_seq - 1, int(elapsed * 30))
            if s["idx"] >= n_seq - 1:
                s["paused"] = True
                s["idx"] = n_seq - 1

        frm = lo + s["idx"]
        ft = force[frm]
        curt = t_rel[frm]
        eef_z_val = eef_z[frm]
        eef_spd = eef_speed[frm]

        # 视频
        if frm in imgs:
            vid_img.set_data(imgs[frm])

        if ft[2] < -20:
            ph, pc = "HEAVY PRESS", "#f85149"
        elif ft[2] < -5:
            ph, pc = "PRESS", "#d29922"
        elif ft[2] > 3:
            ph, pc = "REBOUND", "#3fb950"
        else:
            ph, pc = "NEUTRAL", "#8b949e"

        vid_text.set_text(
            f"F{frm}/{n_total - 1}  t={curt:.2f}s  [{ph}]\n"
            f"Fz={ft[2]:+.1f}N  Fx={ft[0]:+.1f}  Fy={ft[1]:+.1f}\n"
            f"EEF Z={eef_z_val:.3f}m  Spd={eef_spd:.3f}m/s"
        )
        vid_text.set_color(pc)
        vid_text.set_bbox(dict(boxstyle="round", facecolor="black", alpha=0.8,
                               edgecolor=pc, linewidth=2))

        # 柱状图
        ymax = max(abs(ft).max() * 1.3, 3)
        ax_bar.set_ylim(-ymax, ymax)
        for i, (bar, val) in enumerate(zip(bars, ft)):
            bar.set_height(val)
            bar.set_y(min(0, val))
            fmt = f"{val:.3f}" if i >= 3 else f"{val:.1f}"
            bar_lbl[i].set_y(val + ymax * 0.06 if val >= 0 else val - ymax * 0.06)
            bar_lbl[i].set_text(fmt)
            bar_lbl[i].set_va("bottom" if val >= 0 else "top")

        # 所有光标
        for cl in [cursor_eef, cursor_eefv, cursor_fz, cursor_zoom]:
            cl.set_xdata([curt, curt])
        cursor_fz1.set_xdata([curt, curt])
        dot_fz.set_data([curt], [ft[2]])
        dot_zoom.set_data([curt], [ft[2]])

        play_state = "PAUSED" if s["paused"] else f"PLAY {s['speed']:.1f}x"
        status_text.set_text(
            f"F{frm} ({s['idx']+1}/{n_seq})  t={curt:.2f}s  "
            f"Fz={ft[2]:+6.1f}N  EEF Z={eef_z_val:.3f}m  EEF Spd={eef_spd:.3f}m/s  {play_state}  "
            f"|  LR=5f  SHIFT+LR=1f  SPACE=play  UP/DOWN=speed  HOME/END=jump  ESC=quit"
        )

        fig.canvas.draw_idle()

    # ── 键盘 ──
    def on_key(event):
        if event.key is None:
            return
        s = state_dict
        step = 1 if event.key.endswith("shift") is False and hasattr(event, 'key') else 5

        # 实际步长判断用更可靠的方式
        if event.key == "right":
            step_size = 1 if getattr(event, 'guiEvent', None) and hasattr(event.guiEvent, 'modifiers') else 5
            s["idx"] = min(n_seq - 1, s["idx"] + step_size)
            s["t_play_start"] = datetime.datetime.now().timestamp() - s["idx"] / 30.0
        elif event.key == "left":
            step_size = 5
            s["idx"] = max(0, s["idx"] - step_size)
            s["t_play_start"] = datetime.datetime.now().timestamp() - s["idx"] / 30.0
        elif event.key == " ":
            s["paused"] = not s["paused"]
            if not s["paused"]:
                s["t_play_start"] = datetime.datetime.now().timestamp() - s["idx"] / 30.0 / s["speed"]
        elif event.key == "up":
            s["speed"] = min(4.0, s["speed"] + 0.25)
        elif event.key == "down":
            s["speed"] = max(0.25, s["speed"] - 0.25)
        elif event.key == "home":
            s["idx"] = 0
            s["t_play_start"] = datetime.datetime.now().timestamp()
        elif event.key == "end":
            s["idx"] = n_seq - 1
            s["paused"] = True
        elif event.key == "escape":
            plt.close("all")
            return
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)

    state_dict["t_play_start"] = datetime.datetime.now().timestamp()
    update()
    plt.show()


if __name__ == "__main__":
    main()
