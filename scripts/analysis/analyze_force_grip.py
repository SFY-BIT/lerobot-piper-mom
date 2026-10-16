"""对比: 夹住印章(0.029) vs 抓空(0.002) 时的力信号，判断模型能否区分"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DP = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2")

def get_ep(ep_idx):
    for f in DP.rglob(f"episode_{ep_idx:06d}.parquet"):
        t = pq.read_table(f)
        return (np.array(t.column("observation.state").to_pylist()),
                np.array(t.column("observation.force").to_pylist()))
    return None, None

def grip_segments(grip, th=0.031):
    segs, in_c = [], False
    for i, g in enumerate(grip):
        if g < th and not in_c:
            cs = i; in_c = True
        elif g >= th and in_c:
            segs.append((cs, i-1)); in_c = False
    if in_c:
        segs.append((cs, len(grip)-1))
    return segs

# 分类: 正常(grip停在0.029) vs 异常(闭合到0.002以下)
normal, abnormal = [], []
for ep in range(30):
    st, force = get_ep(ep)
    if st is None: continue
    gs = st[:, -1]
    segs = grip_segments(gs)
    if not segs: continue
    for s, e, in segs:
        min_g = gs[s:e+1].min()
        f_hold = force[s:e+1]  # 闭合期间的力
        if min_g < 0.02:
            abnormal.append((ep, s, e, min_g, f_hold))
        else:
            normal.append((ep, s, e, min_g, f_hold))

print(f"正常闭合段(0.029): {len(normal)} | 异常闭合段(<0.02): {len(abnormal)}\n")

print("=== 正常夹住印章时 (state≈0.029) 的力信号 ===")
for ep, s, e, mg, fh in normal[:5]:
    # 闭合刚完成瞬间的力
    f_close = fh[min(20, len(fh)-1)]  # 闭合开始后约20帧
    f_hold = fh[-1] if len(fh) > 50 else fh[-1]
    print(f"ep{ep:>2d} 段[{s}~{e}] min_grip={mg:.3f} | 闭合中力Fz={f_close[2]:+6.2f} 保持力Fz={f_hold[2]:+6.2f}"
          f" | 全力={np.round(f_close,1)}")

print("\n=== 异常抓空时 (state≈0.002) 的力信号 ===")
for ep, s, e, mg, fh in abnormal[:5]:
    f_close = fh[min(20, len(fh)-1)]
    f_hold = fh[-1] if len(fh) > 50 else fh[-1]
    print(f"ep{ep:>2d} 段[{s}~{e}] min_grip={mg:.3f} | 闭合中力Fz={f_close[2]:+6.2f} 保持力Fz={f_hold[2]:+6.2f}"
          f" | 全力={np.round(f_close,1)}")

print()
print("=== 正常闭合段 Fz 的均值统计 ===")
nz = [np.abs(f[:,2].mean()) for _,_,_,_,f in normal]
az = [np.abs(f[:,2].mean()) for _,_,_,_,f in abnormal]
print(f"正常: Fz均值范围 {min(nz):.2f} ~ {max(nz):.2f} (中位 {np.median(nz):.2f})")
print(f"异常: Fz均值范围 {min(az):.2f} ~ {max(az):.2f} (中位 {np.median(az):.2f})")
