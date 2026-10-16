"""分析 16 full: 推理延迟 + 每个 chunk 夹爪轨迹 + 关节动作幅度"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\client_log_20260803_222740.jsonl"
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    queries = [r for r in recs if r.get("query")]
    execs = sorted([r for r in recs if not r.get("query")], key=lambda x: x["step"])

    ts = [q["t"] for q in queries]
    gaps = np.diff(ts)
    print(f"查询帧数: {len(queries)}")
    print(f"推理间隔: mean={gaps.mean():.2f}s median={np.median(gaps):.2f}s (期望 1/3=0.33s)")
    print()

    print("=== 每个 chunk 夹爪 (最后1维) ===")
    for q in queries:
        grip = [s[6] for s in q["chunk_received"]]
        g0, g1 = grip[0], grip[-1]
        trend = "闭合" if g1 < g0 - 0.005 else ("张开" if g1 > g0 + 0.005 else "保持")
        print(f"chunk#{q['chunk_id']:>3d} t={q['t']:6.2f} grip {g0:+.3f}→{g1:+.3f} {trend}")

    print()
    print("=== 每个 chunk 关节动作幅度 (chunk内最大变化) ===")
    for q in queries:
        chunk = np.array(q["chunk_received"])
        span = np.abs(chunk - chunk[0]).max(axis=0)
        print(f"chunk#{q['chunk_id']:>3d} t={q['t']:6.2f} span={np.round(span,3)}")

if __name__ == "__main__":
    main()
