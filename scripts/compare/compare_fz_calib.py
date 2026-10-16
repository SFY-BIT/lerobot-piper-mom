"""对比新标定录制 (erase_calib_check) vs 旧数据集 (erase_board) 的力数据"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

CACHE = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")

def force_stats(name, label, max_ep=30):
    dp = CACHE / name
    if not dp.exists():
        print(f"{label}: 不存在")
        return None
    eps = sorted(dp.rglob("episode_*.parquet"))[:max_ep]
    all_f = []
    starts = []
    for epf in eps:
        t = pq.read_table(epf)
        if "observation.force" not in t.column_names:
            continue
        fc = np.array(t.column("observation.force").to_pylist(), dtype=np.float32)
        all_f.append(fc)
        starts.append(fc[0])
    if not all_f:
        print(f"{label}: 无力数据")
        return None
    f = np.concatenate(all_f)
    s = np.array(starts)
    print(f"\n{'='*60}")
    print(f"{label} ({name}, {len(eps)} ep, {len(f)} 帧)")
    print(f"{'='*60}")
    print(f"{'轴':>3} {'全程mean':>9} {'全程std':>8} {'全程min':>9} {'全程max':>9} {'起始mean':>9}")
    for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        col = f[:, i]
        print(f"{ax:>3} {col.mean():+9.2f} {col.std():8.2f} {col.min():+9.2f} {col.max():+9.2f} {s[:,i].mean():+9.2f}")
    # Fz 分段: 压板段 (Fz < -5) 占比
    fz = f[:, 2]
    press_frac = (fz < -5).mean() * 100
    print(f"\nFz<-5N (压板) 占比: {press_frac:.1f}%")
    return f

def main():
    old = force_stats("erase_board", "旧标定数据集 (8/4 录制)")
    new = force_stats("erase_calib_check", "新标定录制 (8/10, 遥控)")

    if old is not None and new is not None:
        print(f"\n{'='*60}")
        print("Fz 关键对比 (压板时的力)")
        print(f"{'='*60}")
        for label, f in [("旧", old), ("新", new)]:
            fz = f[:, 2]
            press = fz[fz < -5]
            if len(press):
                print(f"{label}: 压板段 Fz mean={press.mean():+8.2f} std={press.std():6.2f} (帧数 {len(press)})")
            else:
                print(f"{label}: 无压板段 (Fz min={fz.min():+.2f})")

if __name__ == "__main__":
    main()
