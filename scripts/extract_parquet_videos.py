#!/usr/bin/env python3
"""从 parquet 内嵌图像 (PNG bytes) 提取合成 mp4 视频。

适用: record_ws_dataset.py 未加 --video 录制 (图像以 PNG 内嵌进 parquet, 无 mp4)。

用法:
  python extract_parquet_videos.py fvt/Pump_bottle_Force_Joint
  python extract_parquet_videos.py fvt/Pump_bottle_Force_Joint --out ./videos_extracted
  python extract_parquet_videos.py fvt/Pump_bottle_Force_Joint --fps 30 --camera one --camera two

输出:
  <out>/camera_one_ep000000.mp4, camera_two_ep000000.mp4 ... (每 episode 每相机一个)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


def main():
    p = argparse.ArgumentParser(description="Extract embedded PNG images to mp4 videos")
    p.add_argument("repo_id", help="数据集名 (如 fvt/Pump_bottle_Force_Joint) 或绝对路径 (如 H:\\a-test\\test\\xxx)")
    p.add_argument("--out", default="", help="输出目录 (默认 <repo>_videos)")
    p.add_argument("--fps", type=int, default=0, help="视频帧率 (默认读 info.json)")
    p.add_argument("--camera", action="append", default=[],
                   help="相机 key 后缀 (one/two), 可多次; 默认全部 (one,two)")
    args = p.parse_args()

    # 支持绝对路径 (H 盘等非 cache 位置) 或 CACHE 下的 repo_id
    if Path(args.repo_id).is_absolute():
        repo_root = Path(args.repo_id)
        repo = repo_root.name
    else:
        repo = args.repo_id.replace("my_robot/", "")
        repo_root = CACHE / repo
    if not repo_root.exists():
        print(f"数据集不存在: {repo_root}")
        return

    info = json.load(open(repo_root / "meta" / "info.json", encoding="utf-8"))
    fps = args.fps or info.get("fps", 30)
    cams = args.camera or ["one", "two"]
    img_keys = [f"observation.images.{c}" for c in cams if f"observation.images.{c}" in info["features"]]
    if not img_keys:
        print(f"数据集无图像特征 (features={list(info['features'])}), 无需提取")
        return

    out = Path(args.out) if args.out else repo_root.parent / f"{Path(repo).name}_videos"
    out.mkdir(parents=True, exist_ok=True)

    eps = sorted(repo_root.rglob("episode_*.parquet"))
    if not eps:
        print(f"无 episode: {repo_root}")
        return

    print(f"数据集: {repo} ({len(eps)} episodes, {fps}fps)")
    for epf in eps:
        ep = int(epf.stem.split("_")[1])
        t = pq.read_table(epf)
        n = len(t)
        for ik in img_keys:
            cam_name = ik.split(".")[-1]
            vpath = out / f"camera_{cam_name}_ep{ep:06d}.mp4"
            writer = cv2.VideoWriter(
                str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
            # 读一帧确定尺寸 (所有帧同尺寸)
            col = t.column(ik)
            w = h = None
            decoded = 0
            for i in range(n):
                struct = col[i].as_py()
                img = cv2.imdecode(np.frombuffer(struct["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if w is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(
                        str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(img)
                decoded += 1
            writer.release()
            print(f"  ep{ep} {cam_name}: {decoded}/{n} 帧 → {vpath.name} ({w}x{h})")
    print(f"\n完成: {out}")


if __name__ == "__main__":
    main()
