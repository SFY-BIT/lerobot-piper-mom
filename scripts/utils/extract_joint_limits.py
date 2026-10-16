"""从数据集提取各关节绝对范围 (min/max) → 保存 JSON 供 client 限幅用"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot" / "my_robot"


def main():
    p = argparse.ArgumentParser(description="Extract joint limits from dataset")
    p.add_argument("repo_id", help="数据集名, 如 my_robot/erase_board_flexiv")
    p.add_argument("--out", default="", help="输出 JSON 路径 (默认 joint_limits_<name>.json)")
    p.add_argument("--margin", type=float, default=0.02,
                   help="额外裕量 (rad), 避免恰好卡边界")
    p.add_argument("--ep-range", default="",
                   help="episode 范围过滤, 如 '0-49' (Erase Board) 或 '50-99' (Pump bottle)。"
                        "空=全部 episode。用于多任务数据集按任务分开提取限幅")
    args = p.parse_args()

    if args.ep_range:
        lo, hi = map(int, args.ep_range.split("-"))
    else:
        lo, hi = 0, 10 ** 9

    dp = CACHE / args.repo_id.replace("my_robot/", "")
    if not dp.exists():
        print(f"数据集不存在: {dp}")
        sys.exit(1)

    eps = sorted(dp.rglob("episode_*.parquet"))
    if not eps:
        print(f"无 episode: {dp}")
        sys.exit(1)

    # 按 episode 范围过滤
    eps = [e for e in eps if lo <= int(e.stem.split("_")[1]) <= hi]
    if not eps:
        print(f"episode 范围 {args.ep_range} 无匹配")
        sys.exit(1)

    # 收集所有帧的 state (7维) 和 action (7维)
    all_state = []
    all_action = []
    for epf in eps:
        t = pq.read_table(epf)
        if "observation.state" in t.column_names:
            st = np.array(t.column("observation.state").to_pylist(), dtype=np.float32)
            all_state.append(st)
        if "action" in t.column_names:
            ac = np.array(t.column("action").to_pylist(), dtype=np.float32)
            all_action.append(ac)

    st = np.concatenate(all_state) if all_state else None
    ac = np.concatenate(all_action) if all_action else None
    print(f"数据集: {args.repo_id} ({len(eps)} episodes)")
    if st is not None:
        print(f"state 帧数: {len(st)}")
    if ac is not None:
        print(f"action 帧数: {len(ac)}")

    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    limits = {"source": str(dp), "episodes": len(eps), "margin_rad": args.margin, "joints": {}}

    print(f"\n{'关节':>8} {'state min':>10} {'state max':>10} {'action min':>11} {'action max':>11} {'限幅范围':>18}")
    for i, name in enumerate(joint_names):
        if st is not None:
            s_min, s_max = float(st[:, i].min()), float(st[:, i].max())
        else:
            s_min, s_max = None, None
        if ac is not None:
            a_min, a_max = float(ac[:, i].min()), float(ac[:, i].max())
        else:
            a_min, a_max = None, None

        # 限幅范围 = state 与 action 的交集 (都覆盖) ± 裕量
        lo = min(x for x in [s_min, a_min] if x is not None) - args.margin
        hi = max(x for x in [s_max, a_max] if x is not None) + args.margin
        limits["joints"][name] = {"min": round(lo, 4), "max": round(hi, 4)}

        s_str = f"{s_min:+.4f}/{s_max:+.4f}" if s_min is not None else "—"
        a_str = f"{a_min:+.4f}/{a_max:+.4f}" if a_min is not None else "—"
        print(f"{name:>8} {s_min if s_min is None else s_min:>10.4f} {s_max if s_max is None else s_max:>10.4f} "
              f"{a_min if a_min is None else a_min:>11.4f} {a_max if a_max is None else a_max:>11.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    out_path = Path(args.out) if args.out else Path(f"joint_limits_{args.repo_id.replace('/', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(limits, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {out_path}")


if __name__ == "__main__":
    main()
