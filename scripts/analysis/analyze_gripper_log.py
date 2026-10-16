"""分析 client_log JSONL: 夹爪闭合/张开动作是否来自同一 chunk"""
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
    print(f"总记录: {len(records)} | 查询帧: {len(queries)} | 执行帧: {len(execs)}")
    print()

    # ── 1. 每个查询帧 chunk 的夹爪轨迹 ──
    print("=" * 70)
    print("1) 每个查询帧收到的 chunk 中夹爪(最后一维) 30 步轨迹")
    print("=" * 70)
    for q in queries:
        grip = [round(s[6], 4) for s in q["chunk_received"]]
        # 紧凑显示: 每 5 步取一个
        grip_c = [grip[i] for i in range(0, 30, 5)]
        mn, mx = min(grip), max(grip)
        print(f"chunk#{q['chunk_id']:>3d} t={q['t']:7.2f} grip30={grip_c}  min={mn:+.3f} max={mx:+.3f} 振幅={mx-mn:.3f}")

    print()
    print("=" * 70)
    print("2) 执行帧实际发出的夹爪指令 (target 最后一维)")
    print("=" * 70)
    # 找出夹爪在阈值内变化的片段
    for e in execs:
        g = e["target"][-1]
        if e["step"] % 5 == 0:  # 每5步打印一次
            print(f"step={e['step']:>4d} t={e['t']:7.2f} chunk#{e['chunk_id']:>3d} grip_target={g:+.4f}")

if __name__ == "__main__":
    main()
