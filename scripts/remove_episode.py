#!/usr/bin/env python3
"""删除数据集中指定 episode，并把后续 episode 前移补位（index 保持连续 0..N-1）。

用于替换坏轨迹的两步流程:
  1) 删除坏轨迹 (例: 删除 ep2, 后续 3~9 自动前移为 2~8, 数据集变 9 条):
       D:\\miniconda\\envs\\lerobot\\python.exe scripts/remove_episode.py \\
         test_board/Erase_Board_FORCE_EEF --index 2
  2) 重新录制 1 条补齐 (resume 从 total_episodes 继续, 新轨迹成为最后一条):
       D:\\miniconda\\envs\\lerobot\\python.exe scripts/record_ws_dataset.py \\
         --server ws://10.96.45.13:8000 --task "Erase Board" \\
         --repo-id test_board/Erase_Board_FORCE_EEF --num-episodes 1 --resume \\
         --episode-time-s 120 --reset-time-s 10 --warmup-time-s 5 \\
         --fps 30 --num-action-steps 30 --show-camera \\
         --start-teleop --record-force-pred --record-value

安全说明 (v2):
  - 原子流程: 先把涉及的所有旧 parquet 改名为唯一临时名 -> 从临时名处理并写最终名
    -> 全部成功后删除临时文件。新/旧文件名永不重叠, 中途失败不丢数据。
    (v1 的"先写新再删旧"因新/旧文件名重叠会误删刚写好的新文件!)
  - 强烈建议 --backup: 先备份 meta + 涉及 parquet 到指定目录。
  - 只支持图像内嵌 parquet 的数据集 (record_ws_dataset.py 默认 use_videos=False)。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

LEROBOT_ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot")


def main() -> None:
    p = argparse.ArgumentParser(description="删除指定 episode 并前移补位")
    p.add_argument("repo_id", help="数据集 repo id, 如 test_board/Erase_Board_FORCE_EEF")
    p.add_argument("--index", type=int, required=True, help="要删除的 episode index")
    p.add_argument("--root", default=str(LEROBOT_ROOT), help="lerobot 缓存根目录")
    p.add_argument("--backup", default="", help="建议: 备份 meta + 涉及 parquet 到该目录")
    p.add_argument("--yes", action="store_true", help="跳过确认提示")
    args = p.parse_args()

    root = Path(args.root) / args.repo_id
    meta_dir = root / "meta"
    chunk_dir = root / "data" / "chunk-000"
    info_p = meta_dir / "info.json"
    if not info_p.exists():
        print(f"数据集不存在: {root}")
        sys.exit(1)

    info = json.load(open(info_p, encoding="utf-8"))
    total_episodes = info["total_episodes"]
    idx = args.index
    if not (0 <= idx < total_episodes):
        print(f"index {idx} 越界 (total_episodes={total_episodes})")
        sys.exit(1)

    # 涉及的所有 parquet: idx (删除) + idx+1..total-1 (前移)
    involved = [ep for ep in range(idx, total_episodes)
                if (chunk_dir / f"episode_{ep:06d}.parquet").exists()]
    if not involved:
        print(f"未找到任何可操作 parquet (index>={idx})")
        sys.exit(1)

    print(f"将删除 ep{idx} 并前移 {[e for e in involved[1:]]} -> "
          f"{[e - 1 for e in involved[1:]]}")
    print(f"涉及 {len(involved)} 个 parquet")

    # ── 0. 备份 (推荐) ──
    if args.backup:
        bak = Path(args.backup)
        print(f"备份 meta + 涉及 parquet 到 {bak} ...")
        shutil.copytree(meta_dir, bak / "meta", dirs_exist_ok=True)
        for ep in involved:
            src = chunk_dir / f"episode_{ep:06d}.parquet"
            if src.exists():
                (bak / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, bak / "data" / "chunk-000" / src.name)
        print("备份完成")

    if not args.yes:
        r = input(f"确认删除 ep{idx} 并前移? [y/N] ").strip().lower()
        if r != "y":
            print("已取消")
            sys.exit(0)

    # ── 1. 全部旧文件改名为唯一临时名 (原子性关键: 新/旧名永不重叠) ──
    tmp_map: dict[int, Path] = {}
    for ep in involved:
        old = chunk_dir / f"episode_{ep:06d}.parquet"
        tmp = chunk_dir / f"_remove_tmp_ep{ep:06d}.parquet"
        old.rename(tmp)
        tmp_map[ep] = tmp
    print("已暂存为临时名:", {ep: t.name for ep, t in tmp_map.items()})

    try:
        # ── 2. 目标 ep: 记录帧数后删除 ──
        removed_frames = len(pd.read_parquet(tmp_map[idx], columns=["index"]))
        tmp_map[idx].unlink()
        print(f"删除 ep{idx} ({removed_frames} 帧)")

        # ── 3. 前移: 从临时名读 -> 写最终名 (ep-1) ──
        for ep in range(idx + 1, total_episodes):
            if ep not in tmp_map:
                continue
            new = chunk_dir / f"episode_{ep - 1:06d}.parquet"
            df = pd.read_parquet(tmp_map[ep])
            n = len(df)
            df["episode_index"] = df["episode_index"] - 1
            df["index"] = df["index"] - removed_frames
            df.to_parquet(new, index=False)
            print(f"前移: episode_{ep:06d}.parquet -> episode_{ep - 1:06d}.parquet ({n} 帧)")
    finally:
        # ── 4. 无论成败都清理临时文件 (成功时新文件已写好) ──
        for tmp in tmp_map.values():
            if tmp.exists():
                tmp.unlink()

    # ── 5. 更新 meta ──
    def shift_jsonl(path: Path) -> None:
        """删除 index==idx 的行, 其余 >idx 的行 episode_index -= 1"""
        if not path.exists():
            return
        lines = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("episode_index") == idx:
                print(f"  meta 删除: {path.name} 中的 ep{idx}")
                continue
            if rec.get("episode_index", -1) > idx:
                rec["episode_index"] -= 1
            lines.append(json.dumps(rec, ensure_ascii=False))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print("更新 meta ...")
    for name in ["episodes.jsonl", "episodes_stats.jsonl", "results.jsonl"]:
        shift_jsonl(meta_dir / name)

    info["total_episodes"] = total_episodes - 1
    info["total_frames"] = info.get("total_frames", 0) - removed_frames
    with open(info_p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"info.json: total_episodes {total_episodes} -> {info['total_episodes']}, "
          f"total_frames -> {info['total_frames']}")

    # ── 6. 校验 ──
    parquet_n = len(list(chunk_dir.glob("episode_*.parquet")))
    ep_jsonl_n = sum(1 for _ in open(meta_dir / "episodes.jsonl", encoding="utf-8") if _.strip())
    print(f"\n完成: parquet={parquet_n}, episodes.jsonl={ep_jsonl_n}, "
          f"total_episodes={info['total_episodes']}")
    if parquet_n != info["total_episodes"] or ep_jsonl_n != info["total_episodes"]:
        print("!!! 校验不一致, 请用 --backup 的备份恢复 !!!")
        sys.exit(1)
    print("校验通过。下一步: 用 --resume --num-episodes 1 录制 1 条补齐。")


if __name__ == "__main__":
    main()
