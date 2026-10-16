"""对比 昨日(8/5) vs 今日(8/6) Erase Board 日志的力数据"""
import json
import numpy as np
from pathlib import Path

LOGS = Path(r"client_logs")

def load_recs(path):
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def force_stats(path, label):
    recs = load_recs(path)
    execs = [r for r in recs if not r.get("query")]
    forces = [e.get("force") for e in execs if e.get("force")]
    if not forces:
        print(f"{label}: 无 force 数据")
        return None
    f = np.array(forces, dtype=np.float32)
    print(f"\n{'='*72}")
    print(f"{label}: {path.name}")
    print(f"执行帧 {len(execs)} | force 帧 {len(f)}")
    print(f"{'轴':>3} {'均值':>8} {'std':>7} {'min':>8} {'max':>8}")
    for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        col = f[:, i]
        print(f"{ax:>3} {col.mean():+8.2f} {col.std():7.2f} {col.min():+8.2f} {col.max():+8.2f}")
    # 力能量 (3轴RMS)
    rms = np.sqrt((f[:, :3] ** 2).sum(axis=1))
    print(f"3轴RMS: 均值={rms.mean():.2f} 峰值={rms.max():.2f}")
    # 高频占比 (Fz)
    from scipy import signal
    try:
        fz = f[:, 2]
        freqs, Pxx = signal.periodogram(fz - fz.mean(), fs=30)
        hi = Pxx[freqs > 2].sum() / Pxx.sum() if Pxx.sum() > 0 else 0
        print(f"Fz 高频占比(>2Hz): {hi:.3f}")
    except Exception:
        pass
    return f

def main():
    print("=" * 72)
    print("昨日 (8/5) Erase Board 日志")
    print("=" * 72)
    y_f = force_stats(LOGS / "SUCCESS_Erase_Board_20260805_155709.jsonl", "成功(success, 旧client)")
    y2_f = force_stats(LOGS / "lora_Erase_Board_20260805_161524.jsonl", "lora(新client)")

    print()
    print("=" * 72)
    print("今日 (8/6) Erase Board 日志")
    print("=" * 72)
    t1 = force_stats(LOGS / "eraseBoard_2w_20260806_143554.jsonl", "2w (旧client 30步)")
    t2 = force_stats(LOGS / "eraseBoard_2w_1Hz_20260806_145325.jsonl", "2w 1Hz (旧client)")

    print()
    print("=" * 72)
    print("对比总结 (均值)")
    print("=" * 72)
    if y_f is not None and t1 is not None:
        for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
            print(f"{ax}: 昨日={y_f[:,i].mean():+7.2f}  今日={t1[:,i].mean():+7.2f}")

if __name__ == "__main__":
    main()
