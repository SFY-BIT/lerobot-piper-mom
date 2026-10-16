"""检查 force_pred: 是否是当前力重复30次 (投影) 还是有真实预测"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\token1_1w_noForce_20260804_160801.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    queries = [r for r in recs if r.get("query")]
    print(f"查询帧: {len(queries)}\n")

    # 检查前几个查询帧
    print("=" * 88)
    print("每个查询帧: force(真实) vs force_pred(预测) 前3行")
    print("=" * 88)
    for q in queries[:6]:
        f_real = q.get("force")
        fp = q.get("force_pred")
        if fp is None:
            print(f"chunk#{q['chunk_id']:>3d}: force_pred = None (服务器没返回预测力!)")
            continue
        fp = np.array(fp)
        f_real = np.array(f_real) if f_real is not None else None
        print(f"chunk#{q['chunk_id']:>3d} force_sent={q.get('force_sent')}")
        if f_real is not None:
            print(f"   真实force: {np.round(f_real, 2)}")
        # 前3行预测
        print(f"   force_pred shape={fp.shape}")
        for i in [0, 1, 2, 15, 29]:
            print(f"    行{i:>2}: {np.round(fp[i], 2)}")

    # ── 关键判断: force_pred 各行是否完全相同 (投影) ──
    print("\n" + "=" * 88)
    print("判断: force_pred 的 30 行是否完全相同 (= 当前力投影)")
    print("=" * 88)
    for q in queries[:10]:
        fp = q.get("force_pred")
        if fp is None:
            continue
        fp = np.array(fp)
        # 各行之间的最大差异
        row_span = np.abs(fp - fp[0]).max() if len(fp) > 0 else 0
        # 30 行是否全等
        identical = np.allclose(fp, fp[0], atol=1e-6)
        print(f"chunk#{q['chunk_id']:>3d}: 30行完全相同={identical}, 行间最大差异={row_span:.4f}")

if __name__ == "__main__":
    main()
