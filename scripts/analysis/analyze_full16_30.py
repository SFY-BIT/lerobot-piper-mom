"""分析 full16k (num-action-steps=30): 是否抖动 + 末端只旋转不平移"""
import json
import sys
import numpy as np
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\full16k_20260803_223309.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]
    execs = sorted([r for r in recs if not r.get("query")], key=lambda x: x["step"])
    queries = [r for r in recs if r.get("query")]
    if not execs:
        print("无执行帧"); return

    states = np.array([e["state"] for e in execs])
    targets = np.array([e["target"] for e in execs])
    times = np.array([e["t"] for e in execs])
    n = len(execs)
    print(f"执行帧 {n} | 查询帧 {len(queries)} | 时间 {times[-1]-times[0]:.1f}s")
    print()

    # ── 1. 各关节运动统计: 位移 + 抖动(高频振荡) ──
    print("=" * 90)
    print("各关节: 位移 + 抖动程度 (相邻帧变化 > 0.005 的频率 = 抖)"
          "  底座关节1-3=位置, 腕部4-6=姿态")
    print("=" * 90)
    print(f"{'关节':>4} {'起点':>7} {'终点':>7} {'净位移':>8} {'总行程':>8} "
          f"{'均步变':>7} {'抖动帧%':>8}  分类")
    for j in range(6):
        s = states[:, j]
        net = s[-1] - s[0]
        travel = np.abs(np.diff(s)).sum()
        step_mean = np.abs(np.diff(s)).mean()
        jitter = (np.abs(np.diff(s)) > 0.005).mean() * 100
        cat = "位置(底座)" if j < 3 else "姿态(腕部)"
        print(f"关节{j+1:<3} {s[0]:+7.3f} {s[-1]:+7.3f} {net:+8.3f} {travel:8.3f} "
              f"{step_mean:7.4f} {jitter:7.1f}%  {cat}")

    # ── 2. 时间序列采样 (每 20 帧) ──
    print()
    print("=" * 90)
    print("关节轨迹采样 (state)")
    print("=" * 90)
    print(f"{'step':>5} {'t':>6} {'j1':>6} {'j2':>6} {'j3':>6} {'j4':>6} {'j5':>6} {'j6':>6}")
    for i in range(0, n, max(1, n // 25)):
        st = states[i]
        print(f"{execs[i]['step']:>5} {times[i]:6.2f} "
              f"{st[0]:+6.3f} {st[1]:+6.3f} {st[2]:+6.3f} {st[3]:+6.3f} {st[4]:+6.3f} {st[5]:+6.3f}")

    # ── 3. 判断: 末端是否平移 (底座关节变化) vs 纯旋转 (腕部变化) ──
    print()
    print("=" * 90)
    print("末端运动推断")
    print("=" * 90)
    base_travel = np.abs(np.diff(states[:, :3], axis=0)).sum()
    wrist_travel = np.abs(np.diff(states[:, 3:6], axis=0)).sum()
    base_net = np.abs(states[-1, :3] - states[0, :3]).sum()
    wrist_net = np.abs(states[-1, 3:6] - states[0, 3:6]).sum()
    print(f"底座关节(1-3) 总行程={base_travel:.3f} rad 净变化={base_net:.3f} rad")
    print(f"腕部关节(4-6) 总行程={wrist_travel:.3f} rad 净变化={wrist_net:.3f} rad")
    if wrist_net > base_net * 2 and base_net < 0.05:
        print("→ 末端基本只旋转(腕部动), 位置几乎不变(底座不动) → 符合'只旋转不平移'")
    elif base_net > 0.05:
        print("→ 末端有平移(底座关节变化)")
    print()

    # ── 4. 夹爪 ──
    g = states[:, -1]
    print(f"夹爪: {g[0]:.3f} → {g[-1]:.3f} (min={g.min():.3f} max={g.max():.3f}) "
          f"闭合过={bool((g<0.03).any())}")

if __name__ == "__main__":
    main()
