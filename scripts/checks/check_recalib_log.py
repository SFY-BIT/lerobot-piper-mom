"""检查重新标定后的日志: 力数据范围 vs 数据集/历史对比"""
import json
import sys
from pathlib import Path
import numpy as np

# 对比基准
DATASET_ERASE = {  # erase_board 数据集初始帧
    "Fx": -1.80, "Fy": +5.30, "Fz": -22.56, "Tx": -0.36, "Ty": +0.16, "Tz": +0.13
}

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\eef_2w2_limited_v2_q4_28_20260810_224413.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    execs = [r for r in recs if not r.get("query")]
    infer_execs = [r for r in execs if r.get("mode") == "infer"]
    teleop_execs = [r for r in execs if r.get("mode") == "teleop"]
    print(f"总记录 {len(recs)} | infer {len(infer_execs)} | teleop {len(teleop_execs)}")

    axes = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    # ── 1. 各阶段力统计 ──
    for label, group in [("infer", infer_execs), ("teleop", teleop_execs)]:
        forces = [r.get("force") for r in group if r.get("force")]
        if not forces:
            print(f"\n{label}: 无力数据")
            continue
        f = np.array(forces, dtype=np.float32)
        print(f"\n=== {label} 阶段力统计 ({len(f)} 帧) ===")
        for i, ax in enumerate(axes):
            col = f[:, i]
            print(f"{ax}: mean={col.mean():+8.2f} std={col.std():6.2f} min={col.min():+8.2f} max={col.max():+8.2f}")

    # ── 2. 与数据集初始帧对比 (擦板任务压板姿态) ──
    if infer_execs:
        f = np.array([r["force"] for r in infer_execs if r.get("force")], dtype=np.float32)
        if len(f):
            print("\n=== 与数据集对比 (擦板压板姿态) ===")
            print(f"{'轴':>3} {'数据集初始':>10} {'本日志mean':>10} {'差异':>8}")
            for i, ax in enumerate(axes):
                ds = DATASET_ERASE[ax]
                mean = f[:, i].mean()
                print(f"{ax:>3} {ds:>+10.2f} {mean:>+10.2f} {mean-ds:>+8.2f}")

    # ── 3. 力传感器噪声 (静置段) ──
    print("\n=== 力噪声 (std 大小判断信号质量) ===")
    for i, ax in enumerate(axes):
        col = f[:, i]
        print(f"{ax}: std={col.std():.3f}")

    # ── 4. 关节范围 (和之前对比) ──
    if infer_execs:
        acts = np.array([r["action"] for r in infer_execs if r.get("action")], dtype=np.float32)
        print("\n=== 关节范围 (infer) ===")
        for i, name in enumerate(["j1","j2","j3","j4","j5","j6","gripper"]):
            col = acts[:, i]
            print(f"{name}: [{col.min():+.3f}, {col.max():+.3f}]")

if __name__ == "__main__":
    main()
