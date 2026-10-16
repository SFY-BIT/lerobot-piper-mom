#!/usr/bin/env python3
"""对比 fvt/Pump_bottle_Force_Joint 与基准数据集的力数值分布, 重点看下压阶段(Fz 负向)。

方法(不用拍脑袋阈值):
  - 下压段 = Fz 相对零位为负 (Fz<0) 且持续 >= 30 帧(1s) 的连续段, 报告各段 min/mean
  - 另给 Fz 直方图 (固定桶宽, 直观对比两数据集负向/正向范围)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_FVT = Path(r"C:\Users\User\.cache\huggingface\lerobot\fvt")
ROOT_MY = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")


def load_force(repo_root: Path, ep_filter=None):
    out = {}
    for p in sorted((repo_root / "data").glob("chunk-*/episode_*.parquet")):
        ep = int(p.stem.split("_")[1])
        if ep_filter and not ep_filter(ep):
            continue
        df = pd.read_parquet(p)
        F = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:6] for x in df["observation.force"]])
        out[ep] = F
    return out


def neg_segments(Fz, min_len=30):
    """Fz<0 的持续段(>=min_len帧), 返回 [(start,end,len,min,mean)]"""
    n = len(Fz)
    mask = Fz < 0
    segs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j - 1, j - i, float(Fz[i:j].min()), float(Fz[i:j].mean())))
            i = j
        else:
            i += 1
    return segs


def hist_bins(all_fz, bins=16, lo=-40, hi=15):
    counts, edges = np.histogram(all_fz, bins=bins, range=(lo, hi))
    return counts, edges


def summarize(repo_root: Path, name: str, ep_filter=None):
    data = load_force(repo_root, ep_filter)
    print(f"\n===== {name} ({len(data)} episodes) =====")
    all_fz = np.concatenate([d[:, 2] for d in data.values()])
    print(f"全部 Fz: n={len(all_fz)}  min={all_fz.min():+.2f}  p1={np.percentile(all_fz,1):+.2f}  "
          f"p5={np.percentile(all_fz,5):+.2f}  mean={all_fz.mean():+.2f}  p95={np.percentile(all_fz,95):+.2f}  "
          f"max={all_fz.max():+.2f}  std={all_fz.std():.2f}")
    print(f"  Fz<0 帧占比: {(all_fz < 0).mean()*100:.1f}%   Fz<-5N: {(all_fz < -5).mean()*100:.1f}%   Fz<-10N: {(all_fz < -10).mean()*100:.1f}%")
    for ep, F in sorted(data.items()):
        Fz = F[:, 2]
        segs = neg_segments(Fz)
        if segs:
            seg_str = "; ".join(f"[{s}-{e}]({l}帧, min={m:+.2f}, mean={a:+.2f})" for s, e, l, m, a in segs[:6])
            if len(segs) > 6:
                seg_str += f" ... 共{len(segs)}段"
        else:
            seg_str = "无持续负向段"
        print(f"  ep{ep} ({len(F)}帧): Fz[min={Fz.min():+.2f} p5={np.percentile(Fz,5):+.2f} "
              f"mean={Fz.mean():+.2f} p95={np.percentile(Fz,95):+.2f} max={Fz.max():+.2f}]")
        print(f"      下压段(Fz<0≥1s): {seg_str}")
    # 直方图 (桶宽 5N, 从 -40 到 +15)
    counts, edges = hist_bins(all_fz)
    print("  Fz 直方图(桶宽5N, [-40,+15]):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / max(1, counts.max()) * 40)
        print(f"    [{lo:+6.1f},{hi:+6.1f}): {c:6d} {bar}")
    return all_fz


if __name__ == "__main__":
    new = summarize(ROOT_FVT / "Pump_bottle_Force_Joint", "fvt/Pump_bottle_Force_Joint (新)")
    base = summarize(ROOT_MY / "total_2task", "total_2task Pump bottle (ep50-99, 基准)",
                     lambda e: 50 <= e <= 99)

    # 直接对比关键指标
    print("\n================ 关键对比 ================")
    print(f"{'指标':<16}{'新(fvt)':>12}{'基准(total_2task)':>22}")
    rows = [
        ("Fz min", new.min(), base.min()),
        ("Fz p5", np.percentile(new, 5), np.percentile(base, 5)),
        ("Fz mean", new.mean(), base.mean()),
        ("Fz max", new.max(), base.max()),
        ("Fz<-5N 占比%", (new < -5).mean() * 100, (base < -5).mean() * 100),
        ("Fz<-10N 占比%", (new < -10).mean() * 100, (base < -10).mean() * 100),
        ("Fz 最大负向幅度", -new[new < 0].min() if (new < 0).any() else 0,
         -base[base < 0].min() if (base < 0).any() else 0),
    ]
    for name, a, b in rows:
        print(f"{name:<16}{a:>12.2f}{b:>22.2f}")
