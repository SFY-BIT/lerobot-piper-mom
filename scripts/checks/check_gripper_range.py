"""检查夹爪范围: 数据集 vs 推理日志"""
import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import sys

def main():
    # ── 1. 数据集夹爪范围 ──
    print("=" * 70)
    print("数据集夹爪范围 (observation.state 最后一维)")
    print("=" * 70)
    CACHE = Path.home() / ".cache" / "huggingface" / "lerobot" / "my_robot"
    for ds in ["stamp_seal_v2", "erase_board"]:
        dp = CACHE / ds
        if not dp.exists():
            continue
        eps = sorted(dp.rglob("episode_*.parquet"))
        grips = []
        for epf in eps[:10]:
            t = pq.read_table(epf)
            st = np.array(t.column("observation.state").to_pylist())
            grips.append(st[:, -1])
        if grips:
            g = np.concatenate(grips)
            print(f"{ds}: min={g.min():.4f} max={g.max():.4f} (范围 {g.max()-g.min():.4f})")

    # ── 2. 最新推理日志夹爪 ──
    print()
    print("=" * 70)
    print("推理日志夹爪 (state / action 最后一维)")
    print("=" * 70)
    logs = sorted(Path("client_logs").glob("eef_*.jsonl")) + sorted(Path("client_logs").glob("*Erase*.jsonl"))
    # 找最新的
    logs = sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    for lp in logs:
        lines = lp.read_text(encoding="utf-8").strip().split("\n")
        recs = [json.loads(l) for l in lines if l.strip()]
        execs = [r for r in recs if not r.get("query")]
        if not execs:
            continue
        st_g = np.array([e["state"][-1] for e in execs if e.get("state")])
        ac_g = np.array([e.get("action", e.get("target"))[-1] for e in execs if e.get("action") or e.get("target")])
        clamps = sum(1 for e in execs if e.get("state"))
        print(f"\n{lp.name}:")
        if len(st_g):
            print(f"  state 夹爪: min={st_g.min():.4f} max={st_g.max():.4f}")
        if len(ac_g):
            print(f"  action 夹爪: min={ac_g.min():.4f} max={ac_g.max():.4f}")

    # ── 3. 限幅参数说明 ──
    print()
    print("=" * 70)
    print("新 client 限幅机制 (clamp_action_delta)")
    print("=" * 70)
    print("""- 限幅是【相对当前值】的增量限幅, 不是绝对范围限制
  gripper 每步最多变化 max_gripper_delta=0.02
  joint    每步最多变化 max_joint_delta=0.2

- 夹爪从 0.01 张到 0.08 (数据集最大范围) 需要:
  (0.08-0.01)/0.02 = 3.5 步 ≈ 117ms @ 30fps → 足够快, 不限制张开

- 真正限制夹爪张开的是【模型输出】本身:
  ⚠️ 如果模型输出的夹爪 target 一直偏小(如 0.02~0.04),
     夹爪就永远到不了 0.08 的张开位置""")

if __name__ == "__main__":
    main()
