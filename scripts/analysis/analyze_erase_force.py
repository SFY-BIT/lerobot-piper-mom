"""分析 Erase Board 日志的力预测质量 (真实力 vs 预测力)"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\Erase_Board_20260805_155709.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    queries = [r for r in recs if r.get("query")]
    execs = sorted([r for r in recs if not r.get("query")], key=lambda x: x["step"])
    print(f"查询帧 {len(queries)} | 执行帧 {len(execs)}")
    print(f"force_sent = {queries[0].get('force_sent') if queries else '?'}")

    # ── 1. 查询帧: 真实力 vs 预测力(首帧) ──
    print("\n" + "=" * 88)
    print("1) 查询帧: 真实 force vs 预测 force (Fz)")
    print("=" * 88)
    rz, pz = [], []
    for q in queries:
        f_real = q.get("force")
        fp = q.get("force_pred")
        if fp is None or f_real is None:
            continue
        f_real = np.array(f_real, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        rz.append(f_real[2])
        pz.append(fp[0, 2])
        if q["chunk_id"] < 8:
            print(f"chunk#{q['chunk_id']:>3} 真实Fz={f_real[2]:+8.2f} 预测Fz首帧={fp[0,2]:+8.2f} "
                  f"预测Fz末帧={fp[-1,2]:+8.2f} | 预测范围={fp[:,2].min():+.1f}~{fp[:,2].max():+.1f}")
    if len(rz) > 2:
        c = np.corrcoef(rz, pz)[0, 1]
        mae = np.mean(np.abs(np.array(rz) - np.array(pz)))
        print(f"\nFz 相关性={c:.3f}, MAE={mae:.2f}N")

    # ── 2. 执行帧: 每帧 真实力 vs 预测力 ──
    print("\n" + "=" * 88)
    print("2) 执行帧: 每帧 真实 force vs 预测 force (Fz, 采样)")
    print("=" * 88)
    rz2, pz2 = [], []
    for i, e in enumerate(execs):
        f = e.get("force")
        fp = e.get("force_pred")
        if f is None or fp is None:
            continue
        rz2.append(f[2])
        pz2.append(fp[2])
    if rz2:
        rz2, pz2 = np.array(rz2), np.array(pz2)
        c2 = np.corrcoef(rz2, pz2)[0, 1]
        mae2 = np.mean(np.abs(rz2 - pz2))
        print(f"逐帧 Fz 相关性={c2:.3f}, MAE={mae2:.2f}N")
        # 采样打印
        for i in range(0, len(execs), max(1, len(execs)//20)):
            e = execs[i]
            f = e.get("force"); fp = e.get("force_pred")
            if f and fp:
                print(f"step={e['step']:>4} t={e['t']:6.2f} 真实Fz={f[2]:+7.2f} 预测Fz={fp[2]:+7.2f}")

    # ── 3. 各轴 MAE ──
    print("\n" + "=" * 88)
    print("3) 各轴 真实 vs 预测 (逐帧) MAE")
    print("=" * 88)
    if len(rz2) > 0:
        for i, ax in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
            r = np.array([e.get("force")[i] for e in execs if e.get("force") and e.get("force_pred")])
            p = np.array([e.get("force_pred")[i] for e in execs if e.get("force") and e.get("force_pred")])
            if len(r) > 0:
                mae = np.mean(np.abs(r - p))
                corr = np.corrcoef(r, p)[0, 1] if np.std(r) > 1e-6 else float('nan')
                bias = np.mean(p - r)
                print(f"{ax}: MAE={mae:6.2f}  相关系数={corr:.3f}  偏差(预测-真实)={bias:+.2f}")

if __name__ == "__main__":
    main()
