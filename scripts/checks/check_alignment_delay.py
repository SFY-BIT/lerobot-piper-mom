"""检查数据集 action 与 state 的对齐: action_i 对应 state_i 还是 state_{i+1}?

验证方法:
  假设 action 是"绝对目标位置" (delta 模型的输入也基于此约定):
  - 若对齐无延迟:  action_i - state_i ≈ state_{i+1} - state_i  (动作在本帧生效)
  - 若 1 帧延迟:  action_i - state_i ≈ state_i - state_i = 0 (动作在下帧才生效)
                 → 应该用 action_{i-1} 匹配 state_i
  - 若 2 帧延迟:  action_i - state_i ≈ state_{i+2} - state_i

通过对比不同延迟下的拟合误差, 找出最小误差的延迟帧数。
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import sys

DP = Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot\stamp_seal_v2")

def main():
    ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    f = next(DP.rglob(f"episode_{ep_idx:06d}.parquet"))
    t = pq.read_table(f)
    state = np.array(t.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.array(t.column("action").to_pylist(), dtype=np.float32)
    n = len(state)
    print(f"episode {ep_idx}: {n} 帧, state={state.shape}, action={action.shape}")
    print(f"采样率假设: 30fps → 帧间隔 33ms\n")

    # 对每个候选延迟 d (帧), 计算误差:
    #   pred_i = state_i + (action_{i} - state_{i})  假设动作立即生效
    #   但实际用 state_{i+d} 作为"动作后的真实状态"
    #   err(d) = mean | (action_i - state_i) - (state_{i+d} - state_i) |
    #          = mean | action_i - state_{i+d} |
    # 找使 action_i ≈ state_{i+d} 的 d → 说明动作在第 d 帧后生效
    print("=" * 78)
    print("候选延迟 d: action_i 与 state_{i+d} 的拟合误差 (越小越匹配)")
    print("=" * 78)
    results = []
    for d in range(-2, 6):
        # action_i 应与 state_{i+d} 匹配
        if d >= 0:
            err = np.mean(np.abs(action[:n-d] - state[d:]))
        else:
            err = np.mean(np.abs(action[-d:] - state[:n+d]))
        results.append((d, err))
        bar = "#" * int(50 * err / max(e for _, e in results))
        print(f"  d={d:+d}  误差={err:.4f}  {bar}")

    best_d = min(results, key=lambda x: x[1])[0]
    print(f"\n→ 最佳延迟: action_i 匹配 state_{'i' if best_d==0 else ('i+%d' % best_d if best_d>0 else 'i%d' % best_d)}")
    if best_d == 0:
        print("  ✅ 无延迟, 帧对齐正常")
    elif best_d == 1:
        print("  ⚠️ 1帧延迟 (action_i 对应 state_{i+1})")
    elif best_d == 2:
        print("  ⚠️ 2帧延迟 (action_i 对应 state_{i+2})")
    else:
        print("  ⚠️ 存在延迟/不对齐")

    # ── 单独看夹爪的对齐 ──
    print("\n" + "=" * 78)
    print("夹爪维度单独分析")
    print("=" * 78)
    sa, ac = state[:, -1], action[:, -1]
    print("  d:  误差 (夹爪)")
    for d in range(-2, 6):
        if d >= 0:
            err = np.mean(np.abs(ac[:n-d] - sa[d:]))
        else:
            err = np.mean(np.abs(ac[-d:] - sa[:n+d]))
        print(f"  d={d:+d}  {err:.4f}")

if __name__ == "__main__":
    main()
