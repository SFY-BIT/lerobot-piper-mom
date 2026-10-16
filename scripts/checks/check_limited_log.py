"""检查日志: 是否加载限幅 + 实际动作是否超出数据集范围"""
import json
import sys
from pathlib import Path
import numpy as np

# 数据集限幅 (从 JSON)
LIMITS = {
    "j1": (-0.2640, 0.9817), "j2": (-0.0344, 2.4314), "j3": (-2.2709, 0.0577),
    "j4": (-0.3968, 0.5215), "j5": (-0.3612, 1.4812), "j6": (-1.3793, 0.9085),
    "gripper": (-0.0200, 0.1000),
}

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"client_logs\eef_2w2_limited_20260810_170137.jsonl"
    print(f"日志: {path}\n")
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(l) for l in lines if l.strip()]

    # 查询帧 (含 chunk_received) + 执行帧 (含 action)
    execs = [r for r in recs if not r.get("query") and r.get("action")]
    infer_execs = [r for r in execs if r.get("mode") == "infer"]
    print(f"总记录 {len(recs)} | 执行帧 {len(execs)} | 推理执行帧 {len(infer_execs)}")
    modes = {}
    for r in recs:
        m = r.get("mode", "?")
        modes[m] = modes.get(m, 0) + 1
    print(f"模式分布: {modes}")

    if not infer_execs:
        print("无推理执行帧!")
        return

    # ── 1. 检查 action 是否超出数据集范围 ──
    print("\n=== 推理执行帧的 action 超范围统计 ===")
    actions = np.array([r["action"] for r in infer_execs], dtype=np.float32)
    names = list(LIMITS.keys())
    for i, name in enumerate(names):
        lo, hi = LIMITS[name]
        col = actions[:, i]
        n_over = int((col > hi).sum())
        n_under = int((col < lo).sum())
        over_vals = col[col > hi]
        under_vals = col[col < lo]
        over_str = f"{over_vals.max():+.4f}" if len(over_vals) else "—"
        under_str = f"{under_vals.min():+.4f}" if len(under_vals) else "—"
        print(f"{name}: 范围[{lo:+.4f},{hi:+.4f}] 实际[{col.min():+.4f},{col.max():+.4f}] "
              f"超上限 {n_over} 帧 (最大 {over_str}) | 低于下限 {n_under} 帧 (最小 {under_str})")

    # ── 2. 超范围帧示例 ──
    print("\n=== 超范围帧示例 (前5) ===")
    shown = 0
    for r in infer_execs:
        a = r["action"]
        over = []
        for i, name in enumerate(names):
            lo, hi = LIMITS[name]
            if a[i] < lo or a[i] > hi:
                over.append(f"{name}={a[i]:+.4f}")
        if over:
            print(f"step={r['step']} t={r['t']:.2f} mode={r.get('mode')} 超范围: {over}")
            shown += 1
            if shown >= 5:
                break

    # ── 3. 检查是否真的加载了限幅 (看是否触发 clamp 日志) ──
    # 日志中无法直接看 clamp warning, 但可以通过查询帧对比 chunk_received vs action
    # 如果限幅生效, action 应 ≈ chunk_received 的钳制后版本
    print("\n=== 限幅是否生效检查 ===")
    queries = [r for r in recs if r.get("query") and r.get("chunk_received")]
    if queries:
        q = queries[0]
        chunk0 = np.array(q["chunk_received"])[0]
        print(f"查询帧0 chunk[0]: {np.round(chunk0, 4)}")
        # 检查有没有明显超范围的 chunk 被钳制
        for i, name in enumerate(names):
            lo, hi = LIMITS[name]
            if chunk0[i] > hi or chunk0[i] < lo:
                print(f"  ⚠️ chunk[0] {name}={chunk0[i]:+.4f} 超出范围 [{lo:+.4f},{hi:+.4f}] (若执行帧被钳, 说明限幅生效)")

if __name__ == "__main__":
    main()
