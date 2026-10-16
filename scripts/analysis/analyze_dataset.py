"""分析录制的数据集：帧率、state/action 对齐、image/state 对齐"""
import pandas as pd
import numpy as np
from pathlib import Path

DATASET = Path(r"C:\Users\User\.cache\huggingface\lerobot\test_alignment\no_gui")
parquet = DATASET / "data" / "chunk-000" / "episode_000000.parquet"
img_dir = DATASET / "data" / "chunk-000"

df = pd.read_parquet(parquet)
print("=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"总帧数: {len(df)}")
print(f"首帧 timestamp: {df['timestamp'].iloc[0]:.4f}s")
print(f"末帧 timestamp: {df['timestamp'].iloc[-1]:.4f}s")
print(f"timestamp 间隔 (理想): {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / (len(df)-1):.6f}s = {1/((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / (len(df)-1)):.1f} hz")
print()

# state 和 action 的对齐
print("=" * 60)
print("state vs action 对齐 (同一帧内)")
print("=" * 60)
state_arr = np.stack(df['observation.state'].apply(lambda x: np.array(x)).tolist())
action_arr = np.stack(df['action'].apply(lambda x: np.array(x)).tolist())
print(f"state shape: {state_arr.shape}")
print(f"action shape: {action_arr.shape}")
diff = np.abs(state_arr - action_arr)
print(f"state-action 绝对差 (前 5 帧, 各关节):")
for i in range(5):
    print(f"  frame {i}: state={state_arr[i].round(4)}  action={action_arr[i].round(4)}  diff={diff[i].round(4)}")
print(f"\nstate-action 全局统计:")
print(f"  平均差: {diff.mean():.4f} rad ({np.degrees(diff.mean()):.2f} deg)")
print(f"  最大差: {diff.max():.4f} rad ({np.degrees(diff.max()):.2f} deg)")
print(f"  中位差: {np.median(diff):.4f} rad ({np.degrees(np.median(diff)):.2f} deg)")
print()

# 帧间 state 变化（看运动连续性）
print("=" * 60)
print("帧间 state 变化 (运动连续性)")
print("=" * 60)
state_dt = np.abs(np.diff(state_arr, axis=0))
print(f"帧间 state 变化 (rad):")
print(f"  平均: {state_dt.mean():.4f} ({np.degrees(state_dt.mean()):.2f} deg/frame)")
print(f"  最大: {state_dt.max():.4f} ({np.degrees(state_dt.max()):.2f} deg/frame)")
print(f"  p95:  {np.percentile(state_dt, 95):.4f} ({np.degrees(np.percentile(state_dt, 95)):.2f} deg/frame)")
print()

# 检查图片文件
print("=" * 60)
print("图片文件检查")
print("=" * 60)
img_one_files = sorted((img_dir / "observation.images.one").glob("*.png")) if (img_dir / "observation.images.one").exists() else []
img_two_files = sorted((img_dir / "observation.images.two").glob("*.png")) if (img_dir / "observation.images.two").exists() else []
print(f"observation.images.one: {len(img_one_files)} 张")
print(f"observation.images.two: {len(img_two_files)} 张")
if img_one_files:
    print(f"  one 首张: {img_one_files[0].name}")
    print(f"  one 末张: {img_one_files[-1].name}")
if img_two_files:
    print(f"  two 首张: {img_two_files[0].name}")
    print(f"  two 末张: {img_two_files[-1].name}")
print()

# 检查 parquet 里的 image 字段
print("=" * 60)
print("parquet image 字段检查")
print("=" * 60)
sample_one = df['observation.images.one'].iloc[0]
sample_two = df['observation.images.two'].iloc[0]
print(f"observation.images.one[0] type: {type(sample_one)}")
print(f"observation.images.one[0] value: {str(sample_one)[:200]}")
print(f"observation.images.two[0] type: {type(sample_two)}")
print(f"observation.images.two[0] value: {str(sample_two)[:200]}")
