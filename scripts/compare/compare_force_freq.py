"""对比 stamp_seal vs erase_board 的力信号高频占比

方法: 对每个 episode 的力通道做 FFT
  高频占比 = 高频段(> cutoff Hz)功率 / 总功率
  采样率 30fps → Nyquist 15Hz
  cutoff 默认 2Hz (按压/擦除的接触力变化通常在 2Hz 以下, 高频=快速冲击/抖动)
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import sys

CACHE = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")

def high_freq_ratio(signal, fps=30, cutoff=2.0):
    """返回高频功率占比 (0~1)"""
    n = len(signal)
    if n < 16:
        return 0.0
    x = signal - signal.mean()
    # 加窗减少频谱泄漏
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(X) ** 2
    mask_hi = freqs > cutoff
    if not mask_hi.any():
        return 0.0
    total = power.sum()
    if total < 1e-12:
        return 0.0
    return power[mask_hi].sum() / total

def analyze(dataset_name, label):
    dp = CACHE / dataset_name
    if not dp.exists():
        print(f"{label}: 数据集不存在 {dp}")
        return None
    ep_files = sorted(dp.rglob("episode_*.parquet"))
    print(f"\n=== {label} ({dataset_name}) — {len(ep_files)} episodes ===")
    ratios = {"Fx": [], "Fy": [], "Fz": [], "Tx": [], "Ty": [], "Tz": []}
    for epf in ep_files[:30]:
        t = pq.read_table(epf)
        if "observation.force" not in t.column_names:
            continue
        fc = np.array(t.column("observation.force").to_pylist(), dtype=np.float32)
        for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
            ratios[ax].append(high_freq_ratio(fc[:, i]))
    if not ratios["Fx"]:
        print("  (无 force 数据)")
        return None
    print(f"{'轴':>3} {'高频占比均值':>10} {'中位':>8} {'std':>7}  (cutoff>2Hz)")
    for ax in ratios:
        r = np.array(ratios[ax])
        print(f"{ax:>3} {r.mean():>10.3f} {np.median(r):>8.3f} {r.std():>7.3f}")
    return ratios

def main():
    stamp = analyze("stamp_seal_v2", "印章任务 stamp_seal_v2")
    erase = analyze("erase_board", "擦板任务 erase_board")
    if stamp and erase:
        print("\n=== 对比 (高频占比均值) ===")
        print(f"{'轴':>3} {'stamp':>8} {'erase':>8} {'erase/stamp':>11}")
        for ax in ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]:
            s = np.array(stamp[ax]).mean()
            e = np.array(erase[ax]).mean()
            r = e / s if s > 0 else float("nan")
            print(f"{ax:>3} {s:>8.3f} {e:>8.3f} {r:>11.2f}")

if __name__ == "__main__":
    main()
