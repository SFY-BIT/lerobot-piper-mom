"""数据集格式转换: piper → flexiv 兼容格式

将我们的 Piper 录制格式转换为参考 flexiv_peel_cucumber_inputForce 的格式:
  1. 合并 observation.force → observation.state (7 → 13 维)
  2. 将外部视频嵌入 parquet (binary bytes)
  3. 可选 float32 → float64

用法:
  python convert_to_flexiv_format.py --input my_robot/stamp_seal --output my_robot/stamp_seal_flexiv
  python convert_to_flexiv_format.py --input my_robot/pick_0722_v2
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_BASE = Path.home() / ".cache" / "huggingface" / "lerobot"


def list_episodes(parquet_dir: Path) -> list[int]:
    """列出所有 episode 编号"""
    episodes = []
    for f in sorted(parquet_dir.glob("episode_*.parquet")):
        ep = int(f.stem.split("_")[1])
        episodes.append(ep)
    return episodes


def find_video_dir(videos_base: Path) -> dict[int, dict[str, Path]]:
    """找到每个 episode 的视频文件路径 (可能有多个摄像头)"""
    video_map = {}
    for cam_dir in videos_base.iterdir():
        if cam_dir.is_dir():
            cam_name = cam_dir.name  # e.g. "observation.images.one"
            for mp4 in cam_dir.glob("*.mp4"):
                ep = int(mp4.stem.split("_")[1])
                if ep not in video_map:
                    video_map[ep] = {}
                video_map[ep][cam_name] = mp4
    return video_map


def load_video_frames(video_path: Path, num_frames: int) -> list[bytes]:
    """从视频中提取所有帧为 JPEG bytes"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(jpg.tobytes())
    cap.release()
    # 如果视频帧数不足，用最后一帧补齐
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else b"")
    return frames


def convert_episode(
    src_parquet: Path,
    dst_parquet: Path,
    video_frames: dict[str, list[bytes]] | None = None,
    embed_images: bool = True,
    use_float64: bool = False,
):
    """转换单个 episode"""
    table = pq.read_table(str(src_parquet))
    n_rows = len(table)
    cols = table.column_names

    # ── 1. 合并 force → state ──
    if "observation.force" in cols:
        force_col = table.column("observation.force")
        state_col = table.column("observation.state")

        # 提取数值
        forces = np.stack([np.array(x) for x in force_col.to_pylist()])
        states = np.stack([np.array(x) for x in state_col.to_pylist()])
        merged = np.concatenate([states, forces], axis=1)  # (N, 7+6=13)

        # 转 float64
        if use_float64:
            merged = merged.astype(np.float64)

        # 构建新的 state 列
        new_state = pa.array([merged[i].tolist() for i in range(n_rows)],
                             type=pa.list_(pa.float64() if use_float64 else pa.float32(), 13))

        # 替换 state 列
        col_idx = cols.index("observation.state")
        new_columns = [table.column(i) for i in range(len(cols))]
        new_columns[col_idx] = new_state

        # 删除 force 列
        force_idx = cols.index("observation.force")
        new_columns.pop(force_idx)
        new_cols = [c for c in cols if c != "observation.force"]
    else:
        new_columns = [table.column(i) for i in range(len(cols))]
        new_cols = cols

    # ── 2. 嵌入图像（如果提供） ──
    if embed_images and video_frames:
        for cam_name, frames in video_frames.items():
            # 创建 struct<bytes, path> 类型
            # 注意: 参考格式的 observation.image 列包含 bytes 和 path
            bytes_list = [pa.scalar(f, type=pa.binary()) for f in frames[:n_rows]]
            # 补齐
            while len(bytes_list) < n_rows:
                bytes_list.append(pa.scalar(b"", type=pa.binary()))

            paths = [f"videos/chunk-000/{cam_name}/episode_{src_parquet.stem}.mp4"] * n_rows

            # 构建 struct 数组
            struct_type = pa.struct([
                ("bytes", pa.binary()),
                ("path", pa.string()),
            ])
            structs = [
                {"bytes": frames[min(i, len(frames) - 1)],
                 "path": paths[min(i, len(paths) - 1)]}
                for i in range(n_rows)
            ]
            img_col = pa.array(structs, type=struct_type)

            # 重命名: observation.images.one → observation.image
            clean_name = cam_name.replace("observation.images.", "observation.image")
            if clean_name == "observation.image.one":
                clean_name = "observation.image"
            elif clean_name == "observation.image.two":
                clean_name = "observation.wrist_image"

            new_columns.append(img_col)
            new_cols.append(clean_name)

    # ── 3. 写新的 parquet ──
    new_table = pa.table({name: col for name, col in zip(new_cols, new_columns)})

    # 更新 schema metadata
    existing_meta = table.schema.metadata
    new_meta = {}
    for k, v in (existing_meta or {}).items():
        if k == b"huggingface":
            hf = json.loads(v.decode("utf-8"))
            # 更新 features
            feats = hf.get("info", {}).get("features", {})
            # 改 state 维度
            if "observation.state" in feats:
                feats["observation.state"]["shape"] = [13]
                feats["observation.state"]["dtype"] = "float64" if use_float64 else "float32"
                feats["observation.state"]["names"] = feats["observation.state"].get("names", []) + [
                    "force_Fx", "force_Fy", "force_Fz", "force_Tx", "force_Ty", "force_Tz"
                ]
            # 删 force key
            feats.pop("observation.force", None)
            # 改 video → image dtype
            for key in list(feats.keys()):
                if "images" in key and feats[key].get("dtype") == "video":
                    feats[key]["dtype"] = "image"
            hf["info"]["features"] = feats
            new_meta[k] = json.dumps(hf).encode("utf-8")
        else:
            new_meta[k] = v

    new_table = new_table.replace_schema_metadata(new_meta)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, str(dst_parquet))
    return n_rows


def convert_dataset(
    src_repo: str,
    dst_repo: str | None = None,
    embed_images: bool = True,
    use_float64: bool = False,
):
    """转换整个数据集"""
    src_base = CACHE_BASE / src_repo
    if dst_repo is None:
        dst_repo = src_repo + "_flexiv"
    dst_base = CACHE_BASE / dst_repo

    if not src_base.exists():
        print(f"ERROR: source not found: {src_base}")
        return

    # 复制 meta 目录
    src_meta = src_base / "meta"
    dst_meta = dst_base / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(src_meta, dst_meta)

    # 数据目录
    src_data = src_base / "data" / "chunk-000"
    dst_data = dst_base / "data" / "chunk-000"
    dst_data.mkdir(parents=True, exist_ok=True)

    episodes = list_episodes(src_data)
    video_map = find_video_dir(src_base / "videos" / "chunk-000") if embed_images else {}

    print(f"Converting {len(episodes)} episodes: {src_repo} → {dst_repo}")
    total_frames = 0
    for ep in sorted(episodes):
        src_pq = src_data / f"episode_{ep:06d}.parquet"
        dst_pq = dst_data / f"episode_{ep:06d}.parquet"

        # 提取视频帧
        vid_frames = {}
        if embed_images and ep in video_map:
            for cam, vpath in video_map[ep].items():
                # 需要知道帧数
                tmp_t = pq.read_table(str(src_pq))
                nf = len(tmp_t)
                vid_frames[cam] = load_video_frames(vpath, nf)
                print(f"  Ep {ep}: loaded {len(vid_frames[cam])} frames from {cam}")

        n = convert_episode(src_pq, dst_pq, vid_frames if vid_frames else None,
                            embed_images, use_float64)
        total_frames += n
        print(f"  Ep {ep}: {n} frames converted")

    # 更新 meta/info.json
    info_path = dst_meta / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    # 更新 features
    feats = info.get("features", {})
    if "observation.state" in feats:
        feats["observation.state"]["shape"] = [13]
        feats["observation.state"]["dtype"] = "float64" if use_float64 else "float32"
        feats["observation.state"]["names"] = feats["observation.state"].get("names", []) + [
            "force_Fx", "force_Fy", "force_Fz", "force_Tx", "force_Ty", "force_Tz"
        ]
    feats.pop("observation.force", None)
    for key in list(feats.keys()):
        if "images" in key and feats[key].get("dtype") == "video":
            feats[key]["dtype"] = "image"
    info["features"] = feats
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDone! {len(episodes)} episodes, {total_frames} frames → {dst_base}")


def main():
    parser = argparse.ArgumentParser(description="Convert Piper dataset to Flexiv-compatible format")
    parser.add_argument("--input", "-i", required=True,
                        help="Source repo (e.g. my_robot/stamp_seal)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output repo (default: input_flexiv)")
    parser.add_argument("--no-embed-images", action="store_true",
                        help="Don't embed images in parquet (faster, smaller)")
    parser.add_argument("--float64", action="store_true",
                        help="Use float64 instead of float32")
    args = parser.parse_args()

    convert_dataset(
        src_repo=args.input,
        dst_repo=args.output,
        embed_images=not args.no_embed_images,
        use_float64=args.float64,
    )


if __name__ == "__main__":
    main()
