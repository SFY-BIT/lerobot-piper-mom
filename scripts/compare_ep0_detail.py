#!/usr/bin/env python3
"""聚焦 ep0 初始位置: 与 total_2task 两任务起始位置逐关节对比 (含角度)。"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
RAD2DEG = 57.2958


def starts_of(repo: str, ep_filter=None):
    out = {}
    for p in sorted((ROOT / repo / "data").glob("chunk-*/episode_*.parquet")):
        ep = int(p.stem.split("_")[1])
        if ep_filter and not ep_filter(ep):
            continue
        df = pd.read_parquet(p)
        out[ep] = np.asarray(df["observation.state"].iloc[0], dtype=np.float32).reshape(-1)[:7]
    return out


def main():
    ep0 = starts_of("Pump_bottle")[0]
    old_pb = starts_of("total_2task", lambda e: e >= 50)   # Pump bottle 任务
    old_eb = starts_of("total_2task", lambda e: e < 50)    # Erase Board 任务
    m_pb = np.array(list(old_pb.values())).mean(0)
    m_eb = np.array(list(old_eb.values())).mean(0)

    print("== ep0 vs total_2task 两个任务起始位置 ==")
    print(f"{'关节':8s} {'ep0':>9s} {'PB均值':>9s} {'ΔPB':>8s} {'ΔPB(°)':>8s} "
          f"{'EB均值':>9s} {'ΔEB':>8s} {'ΔEB(°)':>8s}")
    for i, name in enumerate(JOINTS):
        d_pb = ep0[i] - m_pb[i]
        d_eb = ep0[i] - m_eb[i]
        print(f"{name:8s} {ep0[i]:+9.4f} {m_pb[i]:+9.4f} {d_pb:+8.4f} {d_pb*RAD2DEG:+8.1f} "
              f"{m_eb[i]:+9.4f} {d_eb:+8.4f} {d_eb*RAD2DEG:+8.1f}")

    print(f"\nep0 整体: " + " ".join(f"{n}={v:+.4f}" for n, v in zip(JOINTS, ep0)))
    print(f"ep0 距 PB 均值欧氏距离 = {np.linalg.norm(ep0-m_pb):.4f} rad")
    print(f"ep0 距 EB 均值欧氏距离 = {np.linalg.norm(ep0-m_eb):.4f} rad")

    # total_2task Pump bottle 中与 ep0 最近的 episode
    dists = {ep: np.linalg.norm(s - ep0) for ep, s in old_pb.items()}
    top = sorted(dists.items(), key=lambda x: x[1])[:3]
    print("\ntotal_2task 中 Pump bottle 与 ep0 最接近的 3 个 episode:")
    for ep, d in top:
        s = old_pb[ep]
        print(f"  ep{ep}: 距离={d:.4f}  " + " ".join(f"{n}{v:+.3f}" for n, v in zip(JOINTS, s)))
    print(f"  距离: 中位数={np.median(list(dists.values())):.4f} "
          f"最小={min(dists.values()):.4f} 最大={max(dists.values()):.4f}")

    # 分析 ep0 属于哪个任务的起始簇 (距离 EB 均值更近还是 PB 均值更近)
    print(f"\n归属判断: ep0 距 EB 均值 {np.linalg.norm(ep0-m_eb):.4f} vs 距 PB 均值 {np.linalg.norm(ep0-m_pb):.4f}"
          + (" → 更像 Erase Board 起始" if np.linalg.norm(ep0-m_eb) < np.linalg.norm(ep0-m_pb) else " → 更像 Pump bottle 起始"))


if __name__ == "__main__":
    main()
