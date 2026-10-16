"""检查数据集各 episode 是否有"返回"动作 (起始 vs 结束关节状态)"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DP = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2")

def main():
    print("=== 各 episode 起始/结束关节状态对比 ===")
    print(f"{'ep':>3} {'帧数':>5} {'j1起→末':>12} {'j2起→末':>12} {'j3起→末':>12} {'夹爪起→末':>12} {'末端位移':>8}")
    n_return, n_total = 0, 0
    for epf in sorted(DP.rglob("episode_*.parquet")):
        ep = int(epf.stem.split("_")[1])
        t = pq.read_table(epf)
        st = np.array(t.column("observation.state").to_pylist())
        s0, s1 = st[0], st[-1]
        disp = np.linalg.norm(s1[:6] - s0[:6])
        n_total += 1
        if disp < 0.5:
            ret = "✅返回"
            n_return += 1
        elif disp < 1.0:
            ret = "⚠️部分返回"
        else:
            ret = "❌不返回"
        print(f"{ep:>3} {len(st):>5} {s0[0]:+.2f}→{s1[0]:+.2f} {s0[1]:+.2f}→{s1[1]:+.2f} "
              f"{s0[2]:+.2f}→{s1[2]:+.2f} {s0[6]:.2f}→{s1[6]:.2f} {disp:8.3f} {ret}")
    print(f"\n返回比例: {n_return}/{n_total}")

if __name__ == "__main__":
    main()
