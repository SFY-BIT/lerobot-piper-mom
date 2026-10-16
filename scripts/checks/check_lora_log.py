"""检查 lora_Erase_Board 日志: wrench_history_steps 与 力预测"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\lora_Erase_Board_20260805_161524.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    queries = [r for r in recs if r.get("query")]
    execs = [r for r in recs if not r.get("query")]
    print(f"总记录 {len(recs)} | 查询帧 {len(queries)} | 执行帧 {len(execs)}")

    # ── 1. wrench_history_steps ──
    print("\n=== wrench_history_steps (查询帧) ===")
    if queries:
        steps = [q.get("wrench_history_steps") for q in queries]
        print(f"前15个查询: {steps[:15]}")
        print(f"全部: min={min(steps)} max={max(steps)} 达60帧={[i+1 for i,s in enumerate(steps) if s>=60][:1] or '从未'}")

    # ── 2. force_pred 质量 ──
    print("\n=== 力预测质量 (Fz: 真实 vs 预测) ===")
    rz, pz = [], []
    for e in execs[:20]:
        f = e.get("force")
        fp = e.get("force_pred")
        if f and fp:
            rz.append(f[2]); pz.append(fp[2])
            print(f"step={e['step']:>4} t={e['t']:6.2f} 真实Fz={f[2]:+7.2f} 预测Fz={fp[2]:+7.2f}")

    # 全部执行帧统计
    all_r = np.array([e["force"][2] for e in execs if e.get("force") and e.get("force_pred")])
    all_p = np.array([e["force_pred"][2] for e in execs if e.get("force") and e.get("force_pred")])
    if len(all_r) > 2:
        c = np.corrcoef(all_r, all_p)[0, 1]
        mae = np.mean(np.abs(all_r - all_p))
        bias = np.mean(all_p - all_r)
        print(f"\n全部 {len(all_r)} 帧: corr={c:.3f} MAE={mae:.2f}N 偏差={bias:+.2f}N")

if __name__ == "__main__":
    main()
