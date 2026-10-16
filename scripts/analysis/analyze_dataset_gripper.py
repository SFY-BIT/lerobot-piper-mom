"""分析数据集 stamp_seal_v2 中夹爪 (最后一维) 的闭合/张开模式"""
import numpy as np
import os
import pyarrow.parquet as pq
from pathlib import Path

DP = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2"
CLOSE_TH = 0.03   # 夹爪值 < 0.03 视为"闭合"

def find_segments(grip):
    """找夹爪闭合区间: 返回 [(start_idx, end_idx, min_val), ...]"""
    segs = []
    in_close = False
    for i, g in enumerate(grip):
        if g < CLOSE_TH and not in_close:
            start = i
            in_close = True
        elif g >= CLOSE_TH and in_close:
            segs.append((start, i - 1, float(grip[start:i].min())))
            in_close = False
    if in_close:
        segs.append((start, len(grip) - 1, float(grip[start:].min())))
    return segs

def main():
    data_dir = Path(DP) / "data"
    ep_files = []
    for chunk in sorted(os.listdir(data_dir)):
        cdir = data_dir / chunk
        if not cdir.is_dir():
            continue
        for f in sorted(os.listdir(cdir)):
            if f.startswith("episode_"):
                ep_files.append(cdir / f)
    print(f"共 {len(ep_files)} 个 episode\n")

    for epf in sorted(ep_files):
        ep_idx = int(epf.stem.split("_")[1])
        t = pq.read_table(epf)
        st = np.array(t.column("observation.state").to_pylist())
        ac = np.array(t.column("action").to_pylist())
        grip_s = st[:, -1]
        grip_a = ac[:, -1]
        n = len(grip_s)

        # 夹爪闭合区间 (state)
        segs = find_segments(grip_s)
        # 夹爪张开区间 (state)
        opens = find_segments(1.0 - grip_s)  # 用反向找张开段

        # 统计
        closed_frac = (grip_s < CLOSE_TH).mean() * 100
        grip_min = grip_s.min()
        grip_max = grip_s.max()

        # 找"闭合的固定宽度"：闭合区间内的中位数
        widths = [grip_s[s:e+1].mean() for s, e, m in segs]

        print(f"ep{ep_idx:>2d} 帧数={n:>4d} grip范围=[{grip_min:.3f},{grip_max:.3f}] "
              f"闭合占比={closed_frac:4.1f}% | 闭合段: ", end="")
        if not segs:
            print("无")
        else:
            for s, e, m in segs:
                print(f"[{s}~{e}, {e-s+1}帧, min={m:.3f}] ", end="")
            print()
            print(f"      闭合宽度(各段均值): {[round(w,4) for w in widths]}")

if __name__ == "__main__":
    main()
