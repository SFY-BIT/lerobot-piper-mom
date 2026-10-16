"""对比: with-force vs no-force 的 force_pred (预测力) 质量"""
import json
import sys
import numpy as np
from pathlib import Path

def load(path):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def analyze(path, label):
    print(f"\n{'='*90}")
    print(f"{label}: {Path(path).name}")
    print("="*90)
    recs = load(path)
    queries = [r for r in recs if r.get("query")]
    execs = sorted([r for r in recs if not r.get("query")], key=lambda x: x["step"])
    print(f"查询帧 {len(queries)} | 执行帧 {len(execs)} | force_sent={queries[0].get('force_sent') if queries else '?'}")

    # 对每个查询: 比较 真实力 与 预测力
    print(f"\n{'chunk':>5} {'真实Fz':>8} {'预测Fz行0':>9} {'预测Fz行29':>9} {'预测Fz范围':>12} {'预测/真实':>8}")
    for q in queries[:12]:
        f_real = q.get("force")
        fp = q.get("force_pred")
        if fp is None or f_real is None:
            print(f"chunk#{q['chunk_id']:>3} force_pred=None 或 force=None")
            continue
        f_real = np.array(f_real, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        fz_real = f_real[2]
        fz_pred_0 = fp[0, 2]
        fz_pred_end = fp[-1, 2]
        fz_pred_range = f"{fp[:,2].min():.1f}~{fp[:,2].max():.1f}"
        ratio = (abs(fz_pred_0) / abs(fz_real)) if abs(fz_real) > 0.5 else float('nan')
        print(f"chunk#{q['chunk_id']:>3} {fz_real:+8.2f} {fz_pred_0:+9.2f} {fz_pred_end:+9.2f} {fz_pred_range:>12} {ratio:8.2f}")

    # 相关性: 真实力 vs 预测力(所有执行帧里的真实力 vs 预测力序列)
    # 对每个查询帧, 取该查询时刻附近执行帧的真实力 与 预测力第一帧比较
    corr_x, corr_y = [], []
    for q in queries:
        f_real = q.get("force")
        fp = q.get("force_pred")
        if fp is None or f_real is None:
            continue
        f_real = np.array(f_real, dtype=np.float32)
        fp = np.array(fp, dtype=np.float32)
        corr_x.append(f_real[2])       # 真实 Fz
        corr_y.append(fp[0, 2])        # 预测 Fz 第一帧
    if len(corr_x) > 2:
        c = np.corrcoef(corr_x, corr_y)[0, 1]
        print(f"\n真实Fz 与 预测Fz(首帧) 相关系数 = {c:.3f}")
    else:
        print("\n数据不足, 无法算相关性")

if __name__ == "__main__":
    with_f = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\token1_1w_withForce_20260804_160627.jsonl"
    no_f = sys.argv[2] if len(sys.argv) > 2 else r"client_logs\token1_1w_noForce_20260804_160801.jsonl"
    analyze(with_f, "WITH force")
    analyze(no_f, "NO force (消融)")
