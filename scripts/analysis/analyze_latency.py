"""分析延迟 CSV：send/read 间隔、跟随滞后、实际控制频率"""
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\User\Desktop\piper_latency.csv")
print("=" * 60)
print("延迟日志分析")
print("=" * 60)
print(f"总事件数: {len(df)}")
print(f"事件类型: {df['event_type'].value_counts().to_dict()}")
print()

sends = df[df['event_type'] == 'send'].reset_index(drop=True)
reads = df[df['event_type'] == 'read'].reset_index(drop=True)
print(f"send 数: {len(sends)}, read 数: {len(reads)}")
print()

# send 间隔（控制周期）
send_dt = np.diff(sends['t_s'].values)
print("=" * 60)
print("send 间隔 (控制周期)")
print("=" * 60)
print(f"  平均: {send_dt.mean()*1000:.2f} ms ({1/send_dt.mean():.1f} hz)")
print(f"  中位: {np.median(send_dt)*1000:.2f} ms")
print(f"  最小: {send_dt.min()*1000:.2f} ms")
print(f"  最大: {send_dt.max()*1000:.2f} ms")
print(f"  p95:  {np.percentile(send_dt, 95)*1000:.2f} ms")
print(f"  std:  {send_dt.std()*1000:.2f} ms")
print()

# read 间隔
read_dt = np.diff(reads['t_s'].values)
print("=" * 60)
print("read 间隔")
print("=" * 60)
print(f"  平均: {read_dt.mean()*1000:.2f} ms ({1/read_dt.mean():.1f} hz)")
print(f"  中位: {np.median(read_dt)*1000:.2f} ms")
print(f"  最大: {read_dt.max()*1000:.2f} ms")
print()

# 跟随滞后：send 后多久 read 收敛到目标
print("=" * 60)
print("跟随滞后 (send→read 收敛到 0.02rad)")
print("=" * 60)
THRESH = 0.02
lags = []
send_arr = sends[['j1','j2','j3','j4','j5','j6','gripper']].values
read_arr = reads[['j1','j2','j3','j4','j5','j6','gripper']].values
read_t = reads['t_s'].values
for i in range(len(sends)):
    t_send = sends['t_s'].iloc[i]
    target = send_arr[i]
    # 找 send 之后第一个 read
    mask = read_t > t_send
    if not mask.any():
        continue
    idx_start = np.argmax(mask)
    for j in range(idx_start, len(reads)):
        diff = np.abs(read_arr[j] - target).max()
        if diff < THRESH:
            lags.append(read_t[j] - t_send)
            break
if lags:
    lags = np.array(lags)
    print(f"  样本数: {len(lags)}")
    print(f"  平均: {lags.mean()*1000:.2f} ms")
    print(f"  中位: {np.median(lags)*1000:.2f} ms")
    print(f"  最大: {lags.max()*1000:.2f} ms")
    print(f"  p95:  {np.percentile(lags, 95)*1000:.2f} ms")
print()

# send→下一个 read 间隔
print("=" * 60)
print("send→下一个 read 间隔")
print("=" * 60)
gaps = []
for i in range(len(sends)):
    t_send = sends['t_s'].iloc[i]
    mask = read_t > t_send
    if mask.any():
        idx = np.argmax(mask)
        gaps.append(read_t[idx] - t_send)
gaps = np.array(gaps)
print(f"  平均: {gaps.mean()*1000:.2f} ms")
print(f"  中位: {np.median(gaps)*1000:.2f} ms")
print(f"  最大: {gaps.max()*1000:.2f} ms")
