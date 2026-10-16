"""分析: 机器人卡在初始位置 (16 full 模型) — 看模型输出的动作是否为零/初始位"""
import json
import sys
import numpy as np
from pathlib import Path

def load_records(path):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\client_log_20260803_222740.jsonl"
    print(f"日志: {path}\n")
    records = load_records(path)
    queries = [r for r in records if r.get("query")]
    execs = sorted([r for r in records if not r.get("query")], key=lambda x: x["step"])
    print(f"总记录: {len(records)} | 查询帧: {len(queries)} | 执行帧: {len(execs)}")
    if not queries:
        print("无查询帧!")
        return

    # ── 1. 模型输出的 chunk 是否都≈初始位/零动作 ──
    print("\n" + "=" * 80)
    print("1) 每个查询帧收到的 chunk 动作 (首帧7维) — 判断是否都接近初始位")
    print("=" * 80)
    # 初始状态 (第一个查询帧的 state)
    init_state = np.array(queries[0]["state"])
    print(f"初始 state: {np.round(init_state, 4)}")
    print()
    for q in queries[:12]:
        chunk = np.array(q["chunk_received"])
        first = chunk[0]
        last = chunk[-1]
        # 相对初始 state 的位移
        d_first = np.abs(first - init_state).max()
        d_last = np.abs(last - init_state).max()
        span = np.abs(last - first).max()  # chunk 内部最大变化
        print(f"chunk#{q['chunk_id']:>3d} t={q['t']:6.2f} "
              f"首帧={np.round(first,3)} 尾帧={np.round(last,3)} "
              f"| 首帧距初始={d_first:.3f} 尾帧距初始={d_last:.3f} chunk内span={span:.3f}")

    # ── 2. 执行帧实际 state 变化 (机器人是否动) ──
    print("\n" + "=" * 80)
    print("2) 执行帧的关节状态 (机器人实际有没有动)")
    print("=" * 80)
    n = len(execs)
    if n > 0:
        sample = execs[::max(1, n//15)]
        for e in sample:
            st = np.round(e["state"], 4)
            tg = np.round(e["target"], 4)
            print(f"step={e['step']:>4d} t={e['t']:6.2f} state={st} target={tg}")

if __name__ == "__main__":
    main()
