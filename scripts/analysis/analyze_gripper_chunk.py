"""精确分析夹爪"合上又张开"动作的 chunk 归属"""
import json
import sys
from pathlib import Path

def load_records(path):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\client_log_20260803_220316.jsonl"
    records = load_records(path)
    queries = [r for r in records if r.get("query")]
    execs = [r for r in records if not r.get("query")]

    # 找到夹爪明显闭合/张开的 chunk (振幅 > 0.01)
    print("=== 夹爪动作剧烈的 chunk (振幅>0.01) ===")
    active = []
    for q in queries:
        grip = [s[6] for s in q["chunk_received"]]
        amp = max(grip) - min(grip)
        direction = "闭合" if grip[-1] < grip[0] else ("张开" if grip[-1] > grip[0] else "平")
        if amp > 0.01:
            print(f"chunk#{q['chunk_id']:>3d} t={q['t']:6.2f} 振幅={amp:.3f} {direction}"
                  f"  grip30={[round(g,3) for g in grip]}")
            active.append(q["chunk_id"])

    print()
    print("=== 夹爪动作剧烈时刻的实际执行指令 (相邻 chunk) ===")
    # 找这些 chunk 前后的执行帧
    for q in queries:
        if q["chunk_id"] not in active:
            continue
        cid = q["chunk_id"]
        print(f"\n-- chunk#{cid} (t={q['t']:.2f}) 及其执行帧的 grip_target --")
        # 该 chunk 的执行帧
        for e in execs:
            if e.get("chunk_id") == cid:
                print(f"   step={e['step']:>4d} t={e['t']:6.2f} grip_target={e['target'][-1]:+.4f}")

if __name__ == "__main__":
    main()
