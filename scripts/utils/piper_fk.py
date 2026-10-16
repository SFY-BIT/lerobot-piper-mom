"""Piper 机械臂正运动学 — 从URDF提取

来源: agilexrobotics/agx_arm_urdf/piper/urdf/piper_description.urdf
所有关节为 revolute, axis=(0,0,1)
"""

import numpy as np


def _rz(theta):
    """绕Z轴旋转"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _trans(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])


def _rpy(roll, pitch, yaw):
    """RPY欧拉角转旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.eye(4)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R


def _transform(xyz, rpy_rad):
    """URDF origin: 先平移再旋转"""
    T = _rpy(*rpy_rad)
    T[:3, 3] = xyz
    return T


# ── URDF 提取的关节原点参数 ──
# (x, y, z), (roll, pitch, yaw) — 从 parent link 到 child link
JOINT_ORIGINS = [
    ((0, 0, 0.123), (0, 0, 0)),                              # joint1: base_link → link1
    ((0, 0, 0), (np.pi / 2, -0.1357866, -np.pi)),            # joint2: link1 → link2
    ((0.28503, 0, 0), (0, 0, -1.7938494)),                   # joint3: link2 → link3
    ((-0.02198, -0.25075, 0), (np.pi / 2, 0, 0)),            # joint4: link3 → link4
    ((0, 0, 0), (-np.pi / 2, 0, 0)),                          # joint5: link4 → link5
    ((8.8259e-5, -0.091, 0), (np.pi / 2, 0, 0)),             # joint6: link5 → link6
]

# 末端法兰偏移 (link6末端到法兰)
FLANGE_OFFSET = np.array([0, 0, 0.05])  # 约5cm


def forward_kinematics(joint_angles):
    """
    Piper 正运动学
    Args:
        joint_angles: [q1, q2, q3, q4, q5, q6] 弧度
    Returns:
        pos: (3,) EEF位置 (x, y, z) 米
        R: (3,3) EEF姿态旋转矩阵
    """
    T = np.eye(4)
    for i, q in enumerate(joint_angles):
        xyz, rpy = JOINT_ORIGINS[i]
        T = T @ _transform(xyz, rpy) @ _rz(q)

    # 末端法兰
    pos = T[:3, 3] + T[:3, :3] @ FLANGE_OFFSET
    R = T[:3, :3]

    return pos, R


def compute_eef_trajectory(joint_positions):
    """
    计算整条轨迹的EEF位置
    Args:
        joint_positions: (N, 6) 关节角度
    Returns:
        eef_xyz: (N, 3) EEF位置
        eef_vel: (N,) EEF速度
    """
    n = len(joint_positions)
    eef_xyz = np.zeros((n, 3))
    for i in range(n):
        pos, _ = forward_kinematics(joint_positions[i])
        eef_xyz[i] = pos

    # 速度 (m/s)
    eef_vel = np.zeros(n)
    if n > 1:
        eef_vel[1:] = np.linalg.norm(np.diff(eef_xyz, axis=0), axis=1) / 0.03333
    eef_vel[0] = eef_vel[1] if n > 1 else 0

    return eef_xyz, eef_vel


if __name__ == "__main__":
    # 测试
    import pyarrow.parquet as pq
    parquet = r"C:\Users\User\.cache\huggingface\lerobot\my_robot\pick_0722_v2\data\chunk-000\episode_000000.parquet"
    t = pq.read_table(parquet)
    state = np.stack([np.array(x) for x in t["observation.state"].to_pylist()])
    joints = state[:, :6]

    eef, vel = compute_eef_trajectory(joints)
    print(f"EEF range X: {eef[:,0].min():.3f} ~ {eef[:,0].max():.3f} m")
    print(f"EEF range Y: {eef[:,1].min():.3f} ~ {eef[:,1].max():.3f} m")
    print(f"EEF range Z: {eef[:,2].min():.3f} ~ {eef[:,2].max():.3f} m")
    print(f"EEF speed max: {vel.max():.3f} m/s")
    print(f"First frame EEF: ({eef[0,0]:.3f}, {eef[0,1]:.3f}, {eef[0,2]:.3f})")
    print(f"Impact frame 318 EEF: ({eef[318,0]:.3f}, {eef[318,1]:.3f}, {eef[318,2]:.3f})")
