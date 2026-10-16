"""力传感器+视频同步可视化播放器

播放录制的数据集视频，并实时绘制6轴力/力矩曲线。
支持：暂停/播放、拖拽进度、速度调节。

用法:
    python visualize_force_video.py [--parquet PATH] [--video_dir PATH] [--episode N]

默认使用最新录制的数据集。
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# ── 默认路径 ──────────────────────────────────────────
DEFAULT_PARQUET = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_final\data\chunk-000\episode_000000.parquet"
DEFAULT_VIDEO_DIR = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_final\videos\chunk-000"
DEFAULT_EPISODE = 0


def load_data(parquet_path: str, episode: int = 0):
    """加载指定 episode 的力数据。
    支持两种格式：
    1. 单文件多episode: parquet_path 直接指向 episode_NNNNNN.parquet
    2. 多文件：parquet_path 指向目录，自动找 episode_NNNNNN.parquet
    """
    p = Path(parquet_path)
    if p.is_dir():
        # 目录模式：找对应 episode 的 parquet
        ep_file = p / f"episode_{episode:06d}.parquet"
        if not ep_file.exists():
            raise FileNotFoundError(f"No parquet for episode {episode} in {parquet_path}")
        parquet_path = str(ep_file)
    elif not p.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)

    # 检查是否有 episode_index 列（单文件多episode格式）
    if "episode_index" in table.column_names:
        eps = np.array(table["episode_index"])
        mask = eps == episode
    else:
        # 单episode文件，全部读取
        mask = np.ones(len(table), dtype=bool)

    ts = np.array(table["timestamp"])[mask]
    force = np.stack([np.array(x) for x in table["observation.force"].to_pylist()])[mask]
    frame_idx = np.array(table["frame_index"])[mask]

    return ts, force, frame_idx, mask.sum()


def find_video(video_dir: str, episode: int = 0) -> str | None:
    """查找对应 episode 的 mp4 视频。
    搜索策略：
    1. video_dir/episode_NNNNNN.mp4
    2. video_dir/*/episode_NNNNNN.mp4 (嵌套子目录)
    3. 任意 *.mp4 (回退)
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return None
    pattern = f"episode_{episode:06d}.mp4"
    # 直接匹配
    match = list(video_dir.glob(pattern))
    if match:
        return str(match[0])
    # 递归搜索子目录
    match = list(video_dir.glob(f"**/{pattern}"))
    if match:
        return str(match[0])
    # 回退
    mp4s = list(video_dir.glob("**/*.mp4"))
    if mp4s:
        print(f"WARNING: Using fallback video: {mp4s[0].name}")
        return str(mp4s[0])
    return None


def play_sync(ts, force, video_path, target_fps=30):
    """matplotlib 视频 + 6轴独立柱状图（同步播放）"""
    labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    limits = [(-20, 20), (-20, 20), (-60, 30), (-1.5, 1.5), (-1.5, 1.5), (-0.5, 0.5)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {Path(video_path).name}  ({video_fps:.1f}fps, {total_video_frames} frames)")
    print(f"Force: {len(ts)} frames  ({len(ts)/(ts[-1]-ts[0]):.1f}hz, {ts[-1]-ts[0]:.1f}s)")

    state = {"paused": False, "speed": 1.0, "frame_idx": 0, "t_start": time.perf_counter()}
    data_t0 = ts[0]
    n_frames = len(ts)

    # ── 布局: 3行×2列 柱状图 + 左上方视频 ──
    plt.ion()
    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
    fig.canvas.manager.set_window_title("Force Bar + Video Player — pick_0722_final")

    # 视频区 (左上大图)
    ax_video = fig.add_axes([0.02, 0.40, 0.52, 0.57])
    ax_video.set_facecolor("#000000")
    ax_video.axis("off")
    ret, frame = cap.read()
    im = ax_video.imshow(np.zeros((480, 640, 3), dtype=np.uint8)) if not ret else \
         ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 6个柱状图子图: 2行×3列，放在右侧+下方
    bar_axes = []
    bar_bars = []
    bar_texts = []
    bar_zero_lines = []
    # 位置: [left, bottom, width, height]
    positions = [
        [0.57, 0.70, 0.13, 0.27],  # Fx
        [0.72, 0.70, 0.13, 0.27],  # Fy
        [0.87, 0.70, 0.12, 0.27],  # Fz
        [0.30, 0.07, 0.13, 0.30],  # Tx
        [0.45, 0.07, 0.13, 0.30],  # Ty
        [0.60, 0.07, 0.13, 0.30],  # Tz
    ]

    for idx, (pos, label, color, (ylo, yhi)) in enumerate(zip(positions, labels, colors, limits)):
        ax = fig.add_axes(pos)
        ax.set_facecolor("#16213e")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=3)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, alpha=0.15, color="white", axis="y")
        for spine in ax.spines.values():
            spine.set_color("#333")
        # 零线
        zl = ax.axhline(y=0, color="white", alpha=0.3, linewidth=1)
        # 初始空柱状图 (单柱)
        bar = ax.bar(0, 0, width=0.8, color=color, alpha=0.9, edgecolor="white", linewidth=0.5)[0]
        # 数值标注
        txt = ax.text(0, 0, "", ha="center", va="bottom", color="white",
                      fontsize=9, fontweight="bold", fontfamily="monospace")
        bar_axes.append(ax)
        bar_bars.append(bar)
        bar_texts.append(txt)
        bar_zero_lines.append(zl)

    # 底部信息栏
    info_text = fig.text(0.01, 0.01, "", ha="left", va="bottom",
                         color="#aaa", fontfamily="monospace", fontsize=9)
    ctrl_text = fig.text(0.99, 0.01, "SPACE:Pause  Left/Right:Seek  Up/Down:Speed  ESC:Quit",
                         ha="right", va="bottom", color="#555", fontsize=8)

    print("\n  SPACE=暂停  ←→=跳转±1s  ↑↓=调速  ESC=退出  Q=退出\n")

    def update_frame():
        s = state
        if not s["paused"]:
            elapsed = (time.perf_counter() - s["t_start"]) * s["speed"]
            s["frame_idx"] = min(n_frames - 1, int(elapsed * target_fps))

        fi = s["frame_idx"]
        current_t = ts[fi] - data_t0

        # 视频帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 更新6个柱状图
        vals = force[fi]
        for i, (ax, bar, txt, label, color, (ylo, yhi)) in enumerate(
            zip(bar_axes, bar_bars, bar_texts, labels, colors, limits)
        ):
            v = vals[i]
            # 动态调整y轴范围
            margin = max(abs(ylo), abs(yhi)) * 0.15
            ylo_dyn = min(ylo, v * 1.3 - margin) if v < 0 else ylo
            yhi_dyn = max(yhi, v * 1.3 + margin) if v > 0 else yhi
            ax.set_ylim(ylo_dyn, yhi_dyn)

            # 更新柱高 (set_height改变不了bar的height，需要用patches方式)
            bar.set_height(v)
            if v >= 0:
                bar.set_y(0)
            else:
                bar.set_y(v)

            # 根据正负变色
            if abs(v) < 0.001:
                bar.set_color("#555")
            elif v > 0:
                bar.set_facecolor(color)
            else:
                # 负值用暗红色系
                dark_colors = ["#c0392b", "#27ae60", "#2980b9", "#e67e22", "#8e44ad", "#16a085"]
                bar.set_facecolor(dark_colors[i])

            # 数值标签
            txt.set_y(v + (yhi_dyn - ylo_dyn) * 0.04 if v >= 0 else v - (yhi_dyn - ylo_dyn) * 0.04)
            txt.set_text(f"{v:.2f}" if i <= 2 else f"{v:.3f}")
            txt.set_va("bottom" if v >= 0 else "top")

        status = "PAUSED" if s["paused"] else f'{s["speed"]:.1f}x'
        ax_video.set_title(
            f"Frame {fi}/{n_frames-1}  "
            f"t={current_t:.1f}s  "
            f"{status}",
            color="white" if not s["paused"] else "#f39c12", fontsize=11, loc="left"
        )

        info_text.set_text(
            f"t={current_t:.2f}s  "
            f"Fx={vals[0]:+6.2f}N  Fy={vals[1]:+6.2f}N  Fz={vals[2]:+6.2f}N  "
            f"Tx={vals[3]:+6.3f}Nm  Ty={vals[4]:+6.3f}Nm  Tz={vals[5]:+6.3f}Nm"
        )

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key is None:
            return
        s = state
        fi = s["frame_idx"]
        step = max(1, int(target_fps * 1.0))

        if event.key == " ":
            s["paused"] = not s["paused"]
            if not s["paused"]:
                s["t_start"] = time.perf_counter() - (fi / target_fps) / s["speed"]
        elif event.key in ("escape", "q"):
            plt.close("all")
        elif event.key == "left":
            s["frame_idx"] = max(0, fi - step)
            s["t_start"] = time.perf_counter() - (s["frame_idx"] / target_fps) / s["speed"]
        elif event.key == "right":
            s["frame_idx"] = min(n_frames - 1, fi + step)
            s["t_start"] = time.perf_counter() - (s["frame_idx"] / target_fps) / s["speed"]
        elif event.key == "up":
            s["speed"] = min(4.0, s["speed"] + 0.25)
        elif event.key == "down":
            s["speed"] = max(0.25, s["speed"] - 0.25)
        update_frame()

    fig.canvas.mpl_connect("key_press_event", on_key)

    timer = fig.canvas.new_timer(interval=33)
    timer.add_callback(update_frame)
    timer.start()

    update_frame()
    plt.show(block=True)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Force sensor + video synchronized player",
        epilog="Examples:\n"
               "  python visualize_force_video.py\n"
               "  python visualize_force_video.py --episode 1\n"
               "  python visualize_force_video.py --parquet ./data/chunk-000 --video_dir ./videos/chunk-000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help="Path to parquet file or directory (default: latest recording)")
    parser.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR,
                        help="Path to video directory (searches recursively)")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE,
                        help="Episode number to play")
    parser.add_argument("--fps", type=int, default=0,
                        help="Playback FPS (0=use data timestamps for precise alignment)")
    args = parser.parse_args()

    ts, force, frame_idx, n = load_data(args.parquet, args.episode)
    print(f"Episode {args.episode}: {n} frames, "
          f"duration={ts[-1]-ts[0]:.1f}s, "
          f"fps={n/(ts[-1]-ts[0]):.1f}")

    video_path = find_video(args.video_dir, args.episode)
    if not video_path:
        print("ERROR: No video found!")
        sys.exit(1)

    play_sync(ts, force, video_path, args.fps if args.fps > 0 else 30)


if __name__ == "__main__":
    main()
