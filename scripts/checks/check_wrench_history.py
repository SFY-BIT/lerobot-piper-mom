"""检查各 websocket client 日志: 力历史是否攒到 60 帧"""
import json
from pathlib import Path

def main():
    logs = sorted(Path("client_logs").glob("*.jsonl"))
    print("=== 各日志查询帧的 wrench_history_steps (力历史帧数) ===")
    for lp in logs:
        lines = lp.read_text(encoding="utf-8").strip().split("\n")
        recs = [json.loads(l) for l in lines if l.strip()]
        queries = [r for r in recs if r.get("query")]
        if not queries:
            continue
        if "wrench_history_steps" not in queries[0]:
            print(f"{lp.name}: 无 wrench_history_steps (ZMQ 旧client)")
            continue
        steps = [q.get("wrench_history_steps", 0) for q in queries]
        # 找到第一个 >=60 的位置
        reach = "从未"
        for i, s in enumerate(steps):
            if s >= 60:
                reach = f"第{i+1}个查询"
                break
        print(f"{lp.name}:")
        print(f"  查询帧数={len(queries)} 首帧={steps[0]} 达60帧={reach} 最后={steps[-1]} min={min(steps)}")

if __name__ == "__main__":
    main()
