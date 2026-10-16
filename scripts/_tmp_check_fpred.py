import pyarrow.parquet as pq, numpy as np
from pathlib import Path

d = Path(r'C:\Users\User\.cache\huggingface\lerobot\test\Pump_bottle_Force_Joint')
p = sorted((d / 'data').glob('chunk-*/episode_*.parquet'))[1]
t = pq.read_table(p)

def arr(col):
    a = t.column(col).to_numpy()
    return np.stack(a) if a.dtype == object else a

force = arr('observation.force')
fp = arr('observation.force_pred')
nz = np.where(np.abs(fp).max(axis=1) > 1e-6)[0]
print('推理段: %d ~ %d' % (nz[0], nz[-1]))
i0 = nz[0]
print('=== 连续 15 帧对比 (推理段起始) ===')
print(' 帧 |  force Fx     Fy     Fz  |  pred Fx     Fy     Fz')
for i in range(i0, i0 + 15):
    f, pr = force[i][:3], fp[i][:3]
    print('%3d | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f' % (i, f[0], f[1], f[2], pr[0], pr[1], pr[2]))

print()
print('force 全ep Fx/Fy/Fz 范围:', np.round(force[:, :3].min(axis=0), 2), '~', np.round(force[:, :3].max(axis=0), 2))
print('pred  推理段 Fx/Fy/Fz 范围:', np.round(fp[nz][:, :3].min(axis=0), 2), '~', np.round(fp[nz][:, :3].max(axis=0), 2))
delt = np.diff(force[nz], axis=0)
print('真实力相邻帧差 (若 pred 是 delta 应接近此量级) Fx/Fy/Fz:',
      np.round(np.abs(delt[:, :3]).max(axis=0), 3))
print()
print('=== 判断 ===')
print('force 是传感器读数 = 绝对值 (几 N ~ 几十 N)')
print('pred 量级:', np.round(np.abs(fp[nz]).max(axis=0), 2), '| 若与 force 同量级 → 绝对值; 若接近相邻帧差量级 → delta')
