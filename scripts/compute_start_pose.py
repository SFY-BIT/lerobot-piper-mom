#!/usr/bin/env python3
"""从数据集计算标准初始位姿态: 每个 episode 最初 N 帧 state 平均, 再对全部 episode 平均。

只取轨迹开头 N 帧 (默认 5), 避免把后续运动/漂移帧混进来。
同时验证: 每 episode 前 N 帧内关节是否稳定 (变化量), 防止起点本身在动。
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
RAD2DEG = 57.2958


def main():
    ap = argparse.ArgumentParser(description="计算标准初始位 (最初 N 帧 state 平均)")
    ap.add_argument("--repo", default="total_2task", help="数据集名 (my_robot 下)")
    ap.add_argument("--ep-filter", default="50-99",
                    help="episode 范围过滤, 如 '50-99' 或 '0-49' 或 'all'")
    ap.add_argument("--n-frames", type=int, default=5,
                    help="每个 episode 取最初 N 帧做平均")
    ap.add_argument("--max-eps", type=int, default=0,
                    help="只取前 N 个 episode (0=全部)")
    ap.add_argument("--exclude-eps", default="",
                    help="排除的 episode 列表 (逗号分隔), 如 '51,60,62,63,65,70,72,78'")
    args = ap.parse_args()

    exclude = {int(x) for x in args.exclude_eps.split(",") if x.strip()} if args.exclude_eps else set()

    if args.ep_filter == "all":
        lo, hi = 0, 10 ** 9
    else:
        lo, hi = map(int, args.ep_filter.split("-"))

    # 收集每个 episode 的前 N 帧 state
    per_ep_firstN = {}
    for p in sorted((ROOT / args.repo / "data").glob("chunk-*/episode_*.parquet")):
        ep = int(p.stem.split("_")[1])
        if not (lo <= ep <= hi):
            continue
        if ep in exclude:
            continue
        df = pd.read_parquet(p)
        n = min(args.n_frames, len(df))
        states = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:7] for x in df["observation.state"].iloc[:n]])
        per_ep_firstN[ep] = states
        if args.max_eps and len(per_ep_firstN) >= args.max_eps:
            break

    eps_sorted = sorted(per_ep_firstN)
    print(f"数据集 {args.repo}, episode {args.ep_filter}, 每 ep 取前 {args.n_frames} 帧, "
          f"共 {len(eps_sorted)} 个 episode")

    # 1) 每 episode 前 N 帧稳定性 (帧内最大变化)
    print("\n== 每 episode 前 N 帧稳定性 (首帧→第N帧最大关节变化, 度) ==")
    unstable = []
    for ep in eps_sorted:
        s = per_ep_firstN[ep]
        drift = np.abs(s[-1] - s[0]).max() * RAD2DEG
        flag = "  ⚠ 起点在动!" if drift > 2.0 else ""
        if drift > 2.0:
            unstable.append(ep)
        print(f"  ep{ep:3d}: 首→{args.n_frames}帧最大变化 {drift:6.2f}°{flag}")
    if unstable:
        print(f"  ⚠ {len(unstable)} 个 episode 起点在动: {unstable}")
    else:
        print("  ✓ 所有 episode 起点稳定 (<2°)")

    # 2) 每 episode 前 N 帧平均
    ep_avg = {ep: s.mean(0) for ep, s in per_ep_firstN.items()}

    # 3) 全部 episode 平均 → 标准初始位
    all_avg = np.mean(list(ep_avg.values()), axis=0)
    all_std = np.std(list(ep_avg.values()), axis=0)

    print("\n== 标准初始位 (每 ep 前 N 帧平均 → 全部平均) ==")
    print("  " + " ".join(f"{n}={v:+.4f}" for n, v in zip(JOINTS, all_avg)))
    print(f"  逗号格式 (可直接 --start-pose):")
    print(f"  --start-pose \"{','.join(f'{v:.4f}' for v in all_avg)}\"")
    print(f"  各关节标准差 (越小越一致): " + " ".join(f"{n}={v:.4f}" for n, v in zip(JOINTS, all_std)))
    print(f"  各 ep 平均位 到 标准位 欧氏距离: 中位={np.median([np.linalg.norm(v-all_avg) for v in ep_avg.values()]):.4f} "
          f"最大={max(np.linalg.norm(v-all_avg) for v in ep_avg.values()):.4f}")


if __name__ == "__main__":
    main()
