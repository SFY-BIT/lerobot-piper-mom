"""检查力传感器标定: 当前标定 vs 历史标定 vs 数据集初始状态"""
import numpy as np
import os
import pyarrow.parquet as pq
from pathlib import Path

CAL_DIR = Path(r"C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663")
CACHE = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot")

def main():
    # ── 1. 当前标定文件 ──
    print("=" * 70)
    print("1) 当前标定文件 (FT32663)")
    print("=" * 70)
    base_path = CAL_DIR / "FT32663_tare.npy"
    tool_path = CAL_DIR / "FT32663_tool.npy"
    if base_path.exists():
        base = np.load(base_path)
        print(f"基础偏置 (_tare.npy): {np.round(base, 2)}")
    else:
        print("基础偏置: 文件不存在!")
        base = np.zeros(6)
    if tool_path.exists():
        tool = np.load(tool_path)
        print(f"工具偏置 (_tool.npy): {np.round(tool, 2)}")
    else:
        print("工具偏置: 文件不存在!")
        tool = np.zeros(6)
    print(f"合计偏置: {np.round(base + tool, 2)}")

    # ── 2. 备份的历史标定 (8月3日) ──
    print()
    print("=" * 70)
    print("2) 历史标定对比 (备份)")
    print("=" * 70)
    bak = CAL_DIR / "backup_20260803"
    if bak.exists():
        for f in bak.glob("*.npy"):
            print(f"{f.name}: {np.round(np.load(f), 2)}")
    else:
        print("无备份目录")

    # ── 3. 数据集初始状态 ──
    print()
    print("=" * 70)
    print("3) 数据集各 episode 初始帧 force (录制时起点姿态)")
    print("=" * 70)
    for ds_name in ["stamp_seal_v2", "erase_board"]:
        dp = CACHE / ds_name
        if not dp.exists():
            print(f"\n{ds_name}: 数据集不存在")
            continue
        eps = sorted(dp.rglob("episode_*.parquet"))
        starts = []
        for epf in eps[:12]:
            t = pq.read_table(epf)
            if "observation.force" not in t.column_names:
                continue
            fc = np.array(t.column("observation.force").to_pylist(), dtype=np.float32)
            starts.append(fc[0])
        if starts:
            s = np.array(starts)
            print(f"\n{ds_name} ({len(eps)} ep, 前{len(s)}个初始帧):")
            for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
                print(f"  {ax}: mean={s[:,i].mean():+7.2f}  min={s[:,i].min():+7.2f}  max={s[:,i].max():+7.2f}")

if __name__ == "__main__":
    main()
