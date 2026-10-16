#!/usr/bin/env python3
"""完整读取 total_2task 中 Pump bottle 轨迹 (不预设阈值, 完整看力+关节)。

打印:
  - 每 5 帧一行: 帧号 / 6轴力 / 关键关节(j5,j6,gripper)
  - Fz 的完整曲线 (每 5 帧)
  - 6 轴力极值
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\total_2task")
FX = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def load_ep(ep):
    p = ROOT / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    df = pd.read_parquet(p)
    F = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:6] for x in df["observation.force"]])
    S = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:7] for x in df["observation.state"]])
    return df, F, S


def dump_ep(ep, step=5):
    df, F, S = load_ep(ep)
    n = len(F)
    print(f"\n{'='*100}\n>>> total_2task ep{ep}: {n} 帧 ({n/30:.1f}s)  task={df['task_index'].iloc[0]}\n{'='*100}")
    print(f"力轴顺序: {FX}")
    print(f"关节: j1 j2 j3 j4 j5 j6 gripper")
    print(f"\n{'帧':>4} {'t(s)':>5} | {'Fx':>7} {'Fy':>7} {'Fz':>7} {'Tx':>7} {'Ty':>7} {'Tz':>7} | {'j1':>6} {'j2':>6} {'j3':>6} {'j4':>6} {'j5':>6} {'j6':>6} {'gripper':>7}")
    print("-" * 110)
    for i in range(0, n, step):
        f = F[i]
        s = S[i]
        print(f"{i:>4} {i/30:>5.1f} | {f[0]:>7.2f} {f[1]:>7.2f} {f[2]:>7.2f} {f[3]:>7.2f} {f[4]:>7.2f} {f[5]:>7.2f} | "
              f"{s[0]:>6.3f} {s[1]:>6.3f} {s[2]:>6.3f} {s[3]:>6.3f} {s[4]:>6.3f} {s[5]:>6.3f} {s[6]:>7.3f}")
    # 每轴极值
    print("\n各轴力极值:")
    for k, name in enumerate(FX):
        print(f"  {name}: min={F[:, k].min():+.2f} @帧{np.argmin(F[:, k])}  max={F[:, k].max():+.2f} @帧{np.argmax(F[:, k])}")
    # Fz 深负段 (仅报告 Fz 小于自身 p1 的位置, 看它周围 20 帧的完整力)
    p1 = np.percentile(F[:, 2], 1)
    deep = np.where(F[:, 2] < p1)[0]
    print(f"\nFz 最深区域 (Fz < p1={p1:.2f}, {len(deep)} 帧, 帧号范围 {deep.min()}~{deep.max()} if any):")
    if len(deep):
        c = deep[len(deep) // 2]
        lo, hi = max(0, c - 20), min(n, c + 20)
        print(f"  中心帧 {c} 前后 20 帧完整 6 轴力:")
        print(f"  {'帧':>4} | {'Fx':>7} {'Fy':>7} {'Fz':>7} {'Tx':>7} {'Ty':>7} {'Tz':>7} | {'j5':>6} {'j6':>6} {'gripper':>7}")
        for i in range(lo, hi, 2):
            f, s = F[i], S[i]
            print(f"  {i:>4} | {f[0]:>7.2f} {f[1]:>7.2f} {f[2]:>7.2f} {f[3]:>7.2f} {f[4]:>7.2f} {f[5]:>7.2f} | {s[4]:>6.3f} {s[5]:>6.3f} {s[6]:>7.3f}")


if __name__ == "__main__":
    for ep in [50, 75]:
        dump_ep(ep, step=5)
