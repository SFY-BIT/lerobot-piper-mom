"""检查录制数据集的对齐情况"""
import pyarrow.parquet as pq
import numpy as np

parquet_path = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_final\data\chunk-000\episode_000000.parquet"

t = pq.read_table(parquet_path)
ts = np.array(t["timestamp"])
f = np.stack([np.array(x) for x in t["observation.force"].to_pylist()])
eps = np.array(t["episode_index"])
fi = np.array(t["frame_index"])

print("=" * 60)
print("ALIGNMENT REPORT")
print("=" * 60)
print(f"Total frames:  {len(t)}")
print(f"Duration:      {ts[-1] - ts[0]:.2f}s")
print(f"Overall FPS:   {(len(t) - 1) / (ts[-1] - ts[0]):.1f}")
dt = np.diff(ts)
print(f"Timestamp dt:  min={dt.min():.4f}s  max={dt.max():.4f}s  mean={dt.mean():.4f}s")
print(f"Expected 30hz: 0.0333s")
print()
print("Force/Torque range (6-axis):")
labels = ["Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nm)", "Ty(Nm)", "Tz(Nm)"]
for i, label in enumerate(labels):
    print(f"  {label:8s}  {f[:, i].min():8.3f} ~ {f[:, i].max():8.3f}")
print()

for e in np.unique(eps):
    mask = eps == e
    ets = ts[mask]
    efi = fi[mask]
    ef = f[mask]
    if len(ets) > 1:
        fps = (mask.sum() - 1) / (ets[-1] - ets[0])
    else:
        fps = 0
    print(
        f"Episode {e}: {mask.sum():4d} frames  "
        f"fps={fps:.1f}  "
        f"fi={efi[0]}-{efi[-1]}  "
        f"t={ets[0]:.1f}-{ets[-1]:.1f}s  "
        f"ts_mean={np.diff(ets).mean():.4f}s  "
        f"Fz={ef[:, 2].min():.2f}~{ef[:, 2].max():.2f}N"
    )

print()
print("=" * 60)
print("VERDICT")
overall_fps = (len(t) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0
if 29.5 <= overall_fps <= 30.5 and dt.max() < 0.05:
    print("✓ Alignment is GOOD (30hz stable, no gaps)")
elif dt.max() < 0.1:
    print("⚠ Alignment ACCEPTABLE (minor jitter)")
else:
    print("✗ Alignment POOR (large gaps detected)")
print("=" * 60)
