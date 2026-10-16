"""检查所有带 limit 的日志: infer 阶段 q4 是否超 28° (0.4887 rad)"""
import json
import sys
from pathlib import Path
import numpy as np

Q4_LIMIT_RAD = 0.4887  # 28°
Q4_DATASET_HI = 0.5215  # 数据集上限

def load_recs(path):
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def analyze(path, label=""):
    recs = load_recs(path)
    # 分模式
    infer_execs = [r for r in recs if not r.get("query") and r.get("action") and r.get("mode") == "infer"]
    teleop_execs = [r for r in recs if not r.get("query") and r.get("mode") == "teleop"]
    if not infer_execs:
        print(f"{label}: {path.name} 无 infer 执行帧")
        return
    acts = np.array([r["action"] for r in infer_execs], dtype=np.float32)
    q4 = acts[:, 3]
    n_over_28 = int((q4 > Q4_LIMIT_RAD).sum())
    n_over_ds = int((q4 > Q4_DATASET_HI).sum())
    # 超限的连续段
    over_mask = q4 > Q4_LIMIT_RAD
    segs = []
    in_seg = False
    for i, m in enumerate(over_mask):
        if m and not in_seg:
            s = i; in_seg = True
        elif not m and in_seg:
            segs.append((s, i-1)); in_seg = False
    if in_seg:
        segs.append((s, len(q4)-1))
    print(f"\n{'='*70}")
    print(f"{label}: {path.name}")
    print(f"infer 帧 {len(infer_execs)} | teleop 帧 {len(teleop_execs)}")
    print(f"q4 实际范围: [{np.degrees(q4.min()):.1f}°, {np.degrees(q4.max()):.1f}°]")
    print(f"超 28°: {n_over_28} 帧 ({n_over_28/len(q4)*100:.1f}%) | 超数据集上限 {Q4_DATASET_HI:.4f}: {n_over_ds} 帧")
    if segs:
        print(f"超限段数: {len(segs)}")
        for s, e in segs[:5]:
            print(f"  [{s}~{e}] t={infer_execs[s]['t']:.1f}~{infer_execs[e]['t']:.1f}s "
                  f"q4 max={np.degrees(q4[s:e+1].max()):.1f}°")
    # 检查是否触发 clamp (查询帧与执行帧对比: 若 target 被钳, action 应 ≤ 限幅)
    # 看是否所有超限帧都在 limiter 边界处 (说明 clamp 作用了但被 step-clamp 覆盖)

def main():
    logs_dir = Path("client_logs")
    # 找所有带 limit 的日志
    files = sorted(logs_dir.rglob("*.jsonl"))
    limit_files = [f for f in files if "limited" in f.name or "limit" in f.name]
    if not limit_files:
        print("没找到带 limit 的日志")
        # 列出今天的日志
        today = [f for f in files if "20260810" in f.name]
        print("今天的日志:")
        for f in today:
            print(f"  {f.name}")
        return
    print(f"找到 {len(limit_files)} 个带 limit 的日志:")
    for f in limit_files:
        analyze(f)

if __name__ == "__main__":
    main()
