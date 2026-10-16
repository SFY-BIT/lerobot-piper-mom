"""详细看几个 episode 的夹爪闭合-保持-张开时序 (state vs action)"""
import numpy as np
import os
import pyarrow.parquet as pq
from pathlib import Path

DP = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2")

def get_ep(ep_idx):
    for chunk in sorted(os.listdir(DP / "data")):
        cdir = DP / "data" / chunk
        if not cdir.is_dir():
            continue
        f = cdir / f"episode_{ep_idx:06d}.parquet"
        if f.exists():
            t = pq.read_table(f)
            return np.array(t.column("observation.state").to_pylist()), \
                   np.array(t.column("action").to_pylist())
    return None, None

def print_ep(name, ep_idx, step=10):
    st, ac = get_ep(ep_idx)
    if st is None:
        print(f"{name}: 找不到")
        return
    gs, ga = st[:, -1], ac[:, -1]
    n = len(gs)
    print(f"\n===== {name} (ep{ep_idx}, {n}帧) =====")
    print(f"{'帧':>5} {'state_grip':>10} {'action_grip':>11} {'力Fz':>7}  注")
    # 打印关键帧: 开头、闭合开始、闭合稳定、张开开始、结尾
    keys = [0, 5, 10]
    # 找闭合段和张开段
    close_segs = []
    in_c = False
    for i in range(n):
        if gs[i] < 0.031 and not in_c:
            cs = i; in_c = True
        elif gs[i] >= 0.031 and in_c:
            close_segs.append((cs, i-1)); in_c = False
    if in_c:
        close_segs.append((cs, n-1))
    for s, e in close_segs:
        for i in [s, s+5, s+15, s+30]:
            if i <= e:
                keys.append(i)
        keys.append((s+e)//2)
        keys.append(e)
    # 张开段 (闭合之后)
    for s, e in close_segs:
        if e + 3 < n:
            keys.append(e+1)
            keys.append(e+5)
            keys.append(e+10)
    keys.append(n-1)
    keys = sorted(set(k for k in keys if 0 <= k < n))

    # 读取力
    t2 = pq.read_table(next(DP.rglob(f"episode_{ep_idx:06d}.parquet")))
    force = np.array(t2.column("observation.force").to_pylist())

    prev_s = None
    for i in keys:
        note = ""
        if i < len(close_segs) and close_segs and i == close_segs[0][0]:
            note = "← 闭合开始"
        if prev_s is not None and gs[i] - prev_s > 0.02:
            note = "← 张开跳跃!"
        print(f"{i:>5} {gs[i]:10.4f} {ga[i]:11.4f} {force[i,2]:7.2f}  {note}")
        prev_s = gs[i]

for name, idx in [("ep1(正常)", 1), ("ep12(正常)", 12), ("ep16(异常)", 16), ("ep21(异常)", 21)]:
    print_ep(name, idx)
