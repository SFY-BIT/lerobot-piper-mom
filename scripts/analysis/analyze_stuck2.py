"""分析: 16 full 模型 — 完整轨迹、速度、是否振荡/缓慢"""
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
    execs = sorted([r for r in records if not r.get("query")], key=lambda x: x["step"])
    queries = [r for r in records if r.get("query")]
    if not execs:
        print("无执行帧"); return

    t0 = execs[0]["t"]; t1 = execs[-1]["t"]
    print(f"时间跨度: {t1-t0:.1f}s | 执行帧 {len(execs)} | 查询帧 {len(queries)}")
    print(f"推理频率: {len(queries)/(t1-t0 if t1>t0 else 1):.2f} Hz")

    # ── 各关节轨迹: 起点、终点、最大速度 ──
    states = np.array([e["state"] for e in execs])
    targets = np.array([e["target"] for e in execs])
    times = np.array([e["t"] for e in execs])
    print("\n=== 关节运动总结 ===")
    for j in range(7):
        s0, s1 = states[0, j], states[-1, j]
        vel = np.diff(states[:, j]) / np.diff(times)  # rad/s
        maxv = np.abs(vel).max()
        print(f"关节{j}: {s0:+.3f} → {s1:+.3f} (位移{s1-s0:+.3f}) 最大速度={maxv:.2f} rad/s")

    # ── 位移 vs 时间 (判断是否线性缓慢移动) ──
    print("\n=== 位移随时间 (每2秒采样) ===")
    # 归一化: 相对起点的总位移
    disp = np.linalg.norm(states - states[0], axis=1)
    print(f"{'时间(s)':>8} {'总位移':>8} {'关节0':>7} {'关节4':>7} {'关节5':>7}")
    prev_t = -1
    for i, e in enumerate(execs):
        if e["t"] - prev_t >= 2.0 or i == len(execs)-1:
            d = disp[i]
            print(f"{e['t']:8.2f} {d:8.3f} {states[i,0]:7.3f} {states[i,4]:7.3f} {states[i,5]:7.3f}")
            prev_t = e["t"]

    # ── 夹爪 ──
    g = states[:, -1]
    print(f"\n夹爪: {g[0]:.3f} → {g[-1]:.3f} (min={g.min():.3f} max={g.max():.3f})")
    print(f"夹爪是否闭合过(<0.03): {bool((g<0.03).any())}")

if __name__ == "__main__":
    main()
