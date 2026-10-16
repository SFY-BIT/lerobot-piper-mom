"""检查数据集: 夹爪闭合→保持→松开→返回 的时间线, 及松开时的力信号"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DP = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2")

def main():
    print(f"{'ep':>3} {'帧数':>5} {'闭合开始':>6} {'松开开始':>6} {'保持时长':>6} "
          f"{'松开时Fz':>8} {'闭合时Fz':>8} {'松开时j2':>7} {'结束j2':>6} 说明")
    for epf in sorted(DP.rglob("episode_*.parquet")):
        ep = int(epf.stem.split("_")[1])
        t = pq.read_table(epf)
        st = np.array(t.column("observation.state").to_pylist())
        fc = np.array(t.column("observation.force").to_pylist())
        grip = st[:, -1]
        n = len(grip)
        # 闭合段 (grip < 0.031)
        in_c = False
        for i, g in enumerate(grip):
            if g < 0.031 and not in_c:
                cs = i; in_c = True
            elif g >= 0.031 and in_c:
                ce = i - 1; in_c = False
        if in_c:
            ce = n - 1
        # 松开点 = 闭合段结束后 grip 回升到 >0.04
        release = None
        if in_c or 'ce' in dir():
            for i in range(ce + 1, n):
                if grip[i] > 0.04:
                    release = i
                    break
        if not in_c and 'ce' not in dir():
            continue
        hold_len = ce - cs
        fz_release = fc[release, 2] if release else float("nan")
        fz_close = fc[cs, 2]
        j2_end = st[-1, 1]
        j2_release = st[release, 1] if release else float("nan")
        note = ""
        if release and abs(j2_release - j2_end) > 0.1:
            note = "← 松开后继续返回"
        print(f"{ep:>3} {n:>5} {cs:>6} {str(release):>6} {hold_len:>6} "
              f"{fz_release:>8.1f} {fz_close:>8.1f} {j2_release:>7.2f} {j2_end:>6.2f} {note}")

if __name__ == "__main__":
    main()
