"""按 chunk 划分分析执行动作 + 夹爪闭合/张开模式 (exec-steps=30 完整执行)"""
import json
import sys
from collections import Counter
from pathlib import Path

def load_records(path):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def fmt(x, w=3):
    return f"{x:+.{w}f}"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\client_log_20260803_220820.jsonl"
    records = load_records(path)
    queries = [r for r in records if r.get("query")]
    execs = [r for r in records if not r.get("query")]
    print(f"总记录: {len(records)} | 查询帧: {len(queries)} | 执行帧: {len(execs)}")
    c = Counter(e.get("chunk_id") for e in execs)
    print(f"每 chunk 执行帧数分布: {dict(sorted(c.items()))}")
    print()

    # ── 按 chunk 划分执行动作 (夹爪) ──
    print("=" * 100)
    print("按 chunk 划分: 每个 chunk 实际执行的夹爪轨迹 (首/中/尾)")
    print("=" * 100)
    for q in queries:
        cid = q["chunk_id"]
        chunk_exec = [e for e in execs if e.get("chunk_id") == cid]
        chunk_exec.sort(key=lambda e: e["step"])
        if not chunk_exec:
            continue
        grips = [e["target"][-1] for e in chunk_exec]
        states = [e["state"][-1] for e in chunk_exec]
        n = len(grips)
        def pick(arr, i): return arr[min(i, len(arr)-1)]
        g0, gm, g1 = pick(grips, 0), pick(grips, n//2), pick(grips, n-1)
        s0, s1 = states[0], states[-1]
        trend = "闭合" if g1 < g0 - 0.005 else ("张开" if g1 > g0 + 0.005 else "保持")
        print(f"chunk#{cid:>3d} t={q['t']:6.2f} 帧={n:>2d} {trend}"
              f"  grip目标:{fmt(g0)}→{fmt(gm)}→{fmt(g1)}  实际:{fmt(s0)}→{fmt(s1)}")

    # ── 检测 合→张 瞬间 ──
    print()
    print("=" * 100)
    print("夹爪指令相邻执行帧之间的跳跃 (检测 合→张 瞬间)")
    print("=" * 100)
    prev = None
    for e in sorted(execs, key=lambda x: x["step"]):
        if prev is not None:
            d = e["target"][-1] - prev["target"][-1]
            if abs(d) > 0.01:
                tag = "闭合↓" if d < 0 else "张开↑"
                print(f"step={e['step']:>4d} t={e['t']:6.2f} chunk#{e.get('chunk_id'):>3d}"
                      f"  {tag} {fmt(prev['target'][-1])}→{fmt(e['target'][-1])} (Δ={d:+.3f})")
        prev = e

if __name__ == "__main__":
    main()
