"""检查原始数据集图片存储方式 (video 还是 image)"""
import json
from pathlib import Path

def main():
    dp = Path.home() / ".cache" / "huggingface" / "lerobot" / "my_robot" / "erase_board_test"
    print("=== 原始数据集结构 ===")
    print("顶层:", sorted(p.name for p in dp.iterdir()))

    videos = dp / "videos"
    print("videos 目录存在:", videos.exists())
    if videos.exists():
        print("videos 内容:", sorted(p.name for p in videos.iterdir()))
        chunk0 = videos / "chunk-000"
        if chunk0.exists():
            print("chunk-000 内容(前8):", sorted(p.name for p in chunk0.iterdir())[:8])

    info = json.load(open(dp / "meta" / "info.json"))
    print("\n=== info.json features 图像类型 ===")
    for k, v in info.get("features", {}).items():
        if "image" in k:
            print(f"{k}: dtype={v.get('dtype')}, shape={v.get('shape')}")

if __name__ == "__main__":
    main()
