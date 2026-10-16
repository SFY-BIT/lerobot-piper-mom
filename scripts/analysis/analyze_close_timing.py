"""验证: 推理时夹爪闭合瞬间, 机械臂关节是否还在运动"""
import json
import sys
import numpy as np
from pathlib import Path

def load_records(path):
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]

def main():
    # 取最新日志
    logs = sorted(Path("client_logs").glob("client_log_*.jsonl"))
    path = logs[-1] if logs else None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    print(f"日志: {path}")
    records = load_records(path)
    execs = sorted([r for r in records if not r.get("query")], key=lambda x: x["step"])

    # 找夹爪从张开→闭合的过渡帧
    print("\n=== 夹爪闭合过渡 (grip从>0.05到<0.02) 期间关节运动 ===")
    prev = None
    for e in execs:
        if prev is None:
            prev = e
            continue
        g_prev, g_cur = prev["target"][-1], e["target"][-1]
        # 检测闭合开始 (grip 开始明显下降)
        if g_prev > 0.05 and g_cur < g_prev - 0.005:
            # 打印闭合前 3 帧和闭合过程前几帧的关节速度
            print(f"\n闭合开始: step={e['step']} chunk#{e.get('chunk_id')} t={e['t']:.2f} grip {g_prev:.3f}→{g_cur:.3f}")
            # 找前3帧
            idx = execs.index(e)
            for j in range(max(0, idx-3), min(len(execs), idx+5)):
                ee = execs[j]
                st = np.array(ee["state"][:6])
                tg = np.array(ee["target"][:6])
                diff = np.abs(st - tg)
                arm_motion = np.abs(diff).max()
                print(f"   step={ee['step']:>4d} t={ee['t']:6.2f} grip={ee['target'][-1]:+.4f} "
                      f"关节state={np.round(st,3)} 关节target={np.round(tg,3)} "
                      f"最大|state-target|={arm_motion:.3f}")
        prev = e

if __name__ == "__main__":
    main()
