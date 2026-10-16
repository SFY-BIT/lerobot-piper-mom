#!/usr/bin/env python3
"""比较两个数据集的初始位置 (每个 episode 第一帧 state 7 维)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]


def load_episode_starts(repo: str):
    """返回 {ep_index: 首帧 state[7] (list)}"""
    data = ROOT / repo / "data"
    eps = sorted(data.glob("chunk-*/episode_*.parquet"))
    out = {}
    for p in eps:
        df = pd.read_parquet(p)
        ep = int(p.stem.split("_")[1])
        # state 可能是 list 或 ndarray
        s = df["observation.state"].iloc[0]
        out[ep] = np.asarray(s, dtype=np.float32).reshape(-1)[:7].tolist()
    return out


def summarize(repo: str):
    starts = load_episode_starts(repo)
    arr = np.array(list(starts.values()), dtype=np.float32)
    print(f"\n===== {repo} ({len(starts)} episodes) =====")
    print(f"  首帧 state (每行一个 episode):")
    for ep, s in starts.items():
        print(f"    ep{ep}: " + " ".join(f"{v:+.4f}" for v in s))
    print(f"  均值: " + " ".join(f"{v:+.4f}" for v in arr.mean(axis=0)))
    print(f"  标准差: " + " ".join(f"{v:+.4f}" for v in arr.std(axis=0)))
    print(f"  min: " + " ".join(f"{v:+.4f}" for v in arr.min(axis=0)))
    print(f"  max: " + " ".join(f"{v:+.4f}" for v in arr.max(axis=0)))
    return starts


if __name__ == "__main__":
    repos = sys.argv[1:] or ["Pump_bottle", "total_2task"]
    all_starts = {}
    for r in repos:
        all_starts[r] = summarize(r)

    if len(repos) >= 2:
        a, b = repos[0], repos[1]
        sa = np.array(list(all_starts[a].values()), dtype=np.float32)
        sb = np.array(list(all_starts[b].values()), dtype=np.float32)
        ma, mb = sa.mean(axis=0), sb.mean(axis=0)
        print(f"\n===== 差异 {a} vs {b} (均值差 = {a} - {b}) =====")
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name:8s}: {ma[i]:+.4f} vs {mb[i]:+.4f}  Δ={ma[i]-mb[i]:+.4f} rad ({(ma[i]-mb[i])*57.2958:+.1f}°)")
        # 全局距离
        dist = np.linalg.norm(ma - mb)
        print(f"\n  7维均值向量欧氏距离 = {dist:.4f} (rad)")
        # 每个 episode 到对方均值的距离
        print(f"\n  {a} 各 ep 首帧 → {b} 均值距离:")
        for ep, s in all_starts[a].items():
            d = np.linalg.norm(np.array(s) - mb)
            print(f"    ep{ep}: {d:.4f}")
        print(f"  {b} 各 ep 首帧 → {a} 均值距离:")
        for ep, s in all_starts[b].items():
            d = np.linalg.norm(np.array(s) - ma)
            print(f"    ep{ep}: {d:.4f}")
