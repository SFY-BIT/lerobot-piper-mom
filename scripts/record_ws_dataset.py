#!/usr/bin/env python3
"""Piper 真机数据集录制 — websocket 版 (16 token 模型引导录制)

用 FA-OPENPI 16 token 模型 (serve_policy.py, websocket 8000) 驱动机械臂,
同时把执行过程录制成 lerobot 数据集。

关键设计:
  - 模型输入: 60 帧力历史 (observation/wrench_history [60,6]) — websocket 协议
  - 数据集输入: force 用【当前帧】6维 (observation.force) — 与之前录制格式一致
  - 数据集的 observation.state / action / images 与之前录制完全一致

用法:
  python record_ws_dataset.py --server ws://10.96.45.13:8000 \
      --task "Erase Board" --repo-id my_robot/erase_board_ws \
      --num-episodes 10 --episode-time-s 60 --reset-time-s 10

录制后转换 (可选, 训练前):
  python finalize_dataset.py my_robot/erase_board_ws
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── 键盘监听 (Windows: msvcrt) ──
try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# 按键码 (msvcrt: 方向键/功能键是双字节 b"\xe0"+scancode)
_KEY_LEFT = b"\xe0K"      # ←
_KEY_RIGHT = b"\xe0M"     # →
_KEY_ESC = b"\x1b"
_KEY_H = b"h"             # h: 切换 人工遥控/模型推理
_KEY_S = b"s"             # S: 标注为成功并结束
_KEY_F = b"f"             # F: 标注为失败并结束
_KEY_V = b"v"             # V: 切换 value 标注 0/1 (兼容大写)

# ── 归位慢速参数 (减少电机瞬时负载/发热) ──
RETURN_SPEED = 40           # 归位执行速度 (1-100; 100=最大, 40≈慢速)
RETURN_JOINT_DELTA = 0.06   # 归位每步关节最大变化 (rad, 原 0.2 → 更平滑)
RETURN_GRIPPER_DELTA = 0.02
RETURN_STEPS = 600          # 归位最大插值步数 (20s @30fps, 原 300)
RETURN_TOL = 0.08           # 归位到位容差 (rad; 视觉到位即可, 原 0.02 对 j5 死区太苛刻)
RETURN_STALL_STEPS = 30     # 连续 N 步误差无改善 → 停滞提前退出 (1s @30fps)


def poll_keys() -> list[str]:
    """非阻塞读取按键, 返回事件列表。

    → = 无标注结束(none) | ← = 重置轨迹(丢弃重录) | S = 成功结束 |
    F = 失败结束 | h = 切换 遥控/推理 | V = 切换 value 0/1 | ESC = 完成并退出
    """
    events = []
    if not _HAS_MSVCRT:
        return events
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):          # 方向键/功能键双字节
            ch2 = msvcrt.getch()
            ch = ch + ch2
        if ch == _KEY_RIGHT:
            events.append("done_none")        # → 无标注结束
        elif ch == _KEY_S:
            events.append("done_success")     # S 成功结束
        elif ch == _KEY_F:
            events.append("done_fail")        # F 失败结束
        elif ch == _KEY_LEFT:
            events.append("reset")            # ← 重置轨迹
        elif ch == _KEY_H:
            events.append("toggle")           # h 人工切换
        elif ch in (_KEY_V, b"V"):
            events.append("value")            # V value 标注切换 (大小写兼容)
        elif ch == _KEY_ESC:
            events.append("quit")             # ESC 完成并退出
    return events


def show_cameras(obs: dict, camera_names: list) -> None:
    """实时显示摄像头画面 (人工遥控/录制时观察用)。"""
    if not _HAS_CV2:
        return
    try:
        for name in camera_names:
            img = obs.get(f"observation.images.{name}")
            if img is None:
                continue
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if arr.ndim == 4:
                arr = arr[0]
            cv2.imshow(f"camera {name}", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except Exception as e:
        logging.warning("show_cameras failed: %s", e)


def _lower_speed(robot) -> int:
    """归位前降低电机执行速度, 返回原速度 (供之后恢复)。"""
    old = getattr(robot.arm, "move_speed", 100)
    try:
        robot.arm.set_move_speed(RETURN_SPEED)
    except Exception:
        pass
    return old


def _return_to_target(robot, target: list[float], fps: int,
                      max_joint_delta: float = RETURN_JOINT_DELTA,
                      max_gripper_delta: float = RETURN_GRIPPER_DELTA,
                      tol: float = RETURN_TOL,
                      max_steps: int = RETURN_STEPS,
                      stall_steps: int = RETURN_STALL_STEPS) -> tuple[list[float], float]:
    """低速插值逼近 target; 到位 (<=tol) 或连续 stall_steps 步无改善时提前退出。

    停滞检测: 机械死区/极限下误差无法继续减小 (如 j5), 视觉已到位就视为到位,
    避免卡满 max_steps 时间。返回 (最终 7 维 state, 最终最大误差 rad)。
    """
    target_arr = np.asarray(target[:PROPRIO_DIM], dtype=np.float32)
    best_err = float("inf")
    stall = 0
    final_cur, final_err = None, float("inf")
    for _ in range(max_steps):
        cur = np.asarray(robot.capture_observation()[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
        err = float(np.abs(target_arr - cur).max())
        final_cur, final_err = cur.tolist(), err
        if err <= tol:
            return final_cur, err                 # 到位
        if err < best_err - 1e-4:                 # 有改善 → 重置停滞计数
            best_err, stall = err, 0
        else:
            stall += 1
            if stall >= stall_steps:              # 停滞: 机械死区/极限, 视为已到位
                return final_cur, err
        safe = clamp_action_delta(target_arr.tolist(), cur.tolist(),
                                  max_joint_delta, max_gripper_delta)
        robot.arm.write(safe)
        busy_wait(1.0 / fps)
    return final_cur, final_err


def move_to_home(robot, hold_s: float = 1.5) -> None:
    """归零: 低速插值回启动零位 (避免瞬间全速大行程 → 电机负载/发热)。"""
    log = logging.getLogger("record_ws")
    home = list(robot.arm.init_joint_position)
    home[-1] = robot.startup_gripper_open          # 夹爪张开 (默认 0.1)
    cur0 = np.asarray(robot.capture_observation()[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
    log.info("归零: 起点最大误差 %.3f rad, 低速插值回 home (speed=%d, Δmax=%.3f, tol=%.3f)",
             float(np.abs(home - cur0).max()), RETURN_SPEED, RETURN_JOINT_DELTA, RETURN_TOL)
    old = _lower_speed(robot)
    try:
        cur, err = _return_to_target(robot, home, fps=30)
        if err > RETURN_TOL:
            log.info("归零停滞: 误差 %.3f rad (机械死区/极限, 视觉到位即继续)", err)
        robot.arm.hold_position(home, hold_s=hold_s, interval_s=0.05)
    finally:
        robot.arm.set_move_speed(old)


def replay_to_start(robot, start_state: list[float], fps: int,
                    steps: int = RETURN_STEPS, max_joint_delta: float = RETURN_JOINT_DELTA,
                    max_gripper_delta: float = RETURN_GRIPPER_DELTA,
                    speed: int = RETURN_SPEED, tol: float = RETURN_TOL) -> None:
    """插值回到 episode 起始位置 (作为数据集结束段)。

    从当前 state 线性插值到 start_state, 每步限幅后执行 (低速, 减小电机负载)。
    录制该返回轨迹 → 数据集包含完整闭环: 执行→完成→返回。
    到位容差 tol (默认 0.08 rad, 视觉到位即可); 停滞 (死区/极限) 时提前退出。

    注意: 直接 arm.write (绕过 send_action 的 policy_joint_delta_limit=0.025),
    否则电机对微小增量不响应会卡住 (j5 差 0.055 时卡在 0.129 不动的根因)。
    低速 (speed=RETURN_SPEED) + 小步进 (RETURN_JOINT_DELTA) 减少瞬时负载/发热;
    安全限幅仍由 clamp_action_delta (+ 数据集 JOINT_LIMITS) 保证。
    """
    log = logging.getLogger("record_ws")
    start = np.asarray(start_state[:PROPRIO_DIM], dtype=np.float32)
    cur0 = np.asarray(robot.capture_observation()[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
    log.info("返回段: 起点 %s → 目标 %s (最大误差 %.3f, 低速 speed=%d, tol=%.3f)",
             [f"{v:+.3f}" for v in cur0], [f"{v:+.3f}" for v in start],
             float(np.abs(start - cur0).max()), speed, tol)
    old = _lower_speed(robot)
    try:
        cur, err = _return_to_target(robot, start_state, fps,
                                     max_joint_delta, max_gripper_delta,
                                     tol=tol, max_steps=steps)
    finally:
        robot.arm.set_move_speed(old)
    log.info("返回段结束: 最终 %s (误差 %.3f rad)%s",
             [f"{v:+.3f}" for v in cur], err,
             " ✓" if err <= tol else " ⚠ 未到位")


# ── openpi-client 的 websocket + msgpack 编码 ──
def _find_openpi_client_src() -> Path:
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        cand = parent / "packages" / "src"
        if (cand / "openpi_client").is_dir():
            return cand
    return Path(__file__).resolve().parent.parent / "packages" / "src"


_OPENPI_CLIENT_SRC = _find_openpi_client_src()
sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

import websockets.sync.client
from openpi_client import msgpack_numpy

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait

# =========================================================================
# 配置常量 — 与 piper_ws_force_client.py 一致
# =========================================================================
FT_HISTORY_STEPS = 60      # 力历史帧数 (发送给模型的)
FORCE_DIM = 6
PROPRIO_DIM = 7
ACTION_DIM = 7
CAMERA_MAIN = "observation.images.one"
CAMERA_WRIST = "observation.images.two"
STATE_KEY = "observation.state"
FORCE_KEY = "observation.force"   # 数据集/真机 当前帧力


def build_wrench_history(force_buf: deque) -> np.ndarray:
    frames = list(force_buf)
    T_avail = len(frames)
    if T_avail == 0:
        # force_buf 为空 (如 --zero-force-to-server 消融): 直接返回全 0 [60,6]
        return np.zeros((FT_HISTORY_STEPS, FORCE_DIM), dtype=np.float32)
    if T_avail >= FT_HISTORY_STEPS:
        return np.stack(frames[-FT_HISTORY_STEPS:], axis=0)
    pad = np.zeros((FT_HISTORY_STEPS - T_avail, FORCE_DIM), dtype=np.float32)
    real = np.stack(frames, axis=0)
    return np.concatenate([pad, real], axis=0)


def build_observation(obs: dict[str, Any], force_buf: deque, prompt: str) -> dict[str, Any]:
    img = _to_uint8(obs[CAMERA_MAIN])
    wrist = _to_uint8(obs[CAMERA_WRIST])
    state = np.asarray(obs[STATE_KEY], dtype=np.float32)
    if state.ndim > 1:
        state = state[0]
    proprio = state[:PROPRIO_DIM]
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": proprio.tolist(),
        "observation/wrench_history": build_wrench_history(force_buf),
        "prompt": prompt,
    }


def _to_uint8(img: Any) -> np.ndarray:
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def clamp_action_delta(target: list[float], current: list[float],
                       max_joint_delta: float, max_gripper_delta: float) -> list[float]:
    clamped = list(target)
    for i, (t, c) in enumerate(zip(target, current)):
        max_d = max_gripper_delta if i == len(target) - 1 else max_joint_delta
        lo, hi = c - max_d, c + max_d
        clamped[i] = max(lo, min(hi, float(t)))
    # 数据集绝对限幅 (若已加载): 防止动作超出数据集范围
    if JOINT_LIMITS is not None:
        for idx, (lo, hi) in JOINT_LIMITS.items():
            if idx < len(clamped):
                clamped[idx] = max(lo, min(hi, clamped[idx]))
    return clamped


# 数据集关节绝对限幅 {idx: (min, max)} (由 --joint-limits-file 加载)
JOINT_LIMITS: dict[int, tuple[float, float]] | None = None


def main() -> None:
    p = argparse.ArgumentParser(description="Piper websocket dataset recorder (16 token model)")
    p.add_argument("--server", default="ws://127.0.0.1:8000")
    p.add_argument("--no-server", action="store_true",
                   help="纯本地手控录制: 不连接 server, 不推理 (需配合 --start-teleop 全程遥控)")
    p.add_argument("--task", default="Erase Board", help="任务指令 (prompt)")
    p.add_argument("--repo-id", required=True, help="数据集名, 如 my_robot/erase_board_ws")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--warmup-time-s", type=float, default=5.0)
    p.add_argument("--reset-time-s", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--num-action-steps", type=int, default=30,
                   help="每次推理后执行的 chunk 步数 (1~30)")
    p.add_argument("--max-joint-delta", type=float, default=0.2)
    p.add_argument("--max-gripper-delta", type=float, default=0.02)
    p.add_argument("--robot-type", default="piper")
    p.add_argument("--video", action="store_true", help="录制视频 (需要 ffmpeg)")
    p.add_argument("--resume", action="store_true", help="继续已有数据集")
    p.add_argument("--no-force", action="store_true",
                   help="不发送力给模型 (消融, 模型输入全0); 数据集仍记录当前帧力")
    p.add_argument("--zero-force-to-server", action="store_true",
                   help="发送给 server 的力全 0 (模型输入消融), 数据集仍记录本地真实力,"
                        "force_pred 正常记录 (配合 --record-force-pred)。"
                        "与 --no-force 等价, 语义更明确")
    p.add_argument("--record-force-pred", action="store_true",
                   help="记录模型预测力 observation.force_pred (6维, 当前步对应行) 到数据集。"
                        "推理帧=模型预测, 遥控/返回段=零; 关闭则不生成该特征")
    p.add_argument("--record-value", action="store_true",
                   help="记录 V 键切换的 value 标注 observation.value (int64, 每帧一个 0/1) 到数据集。"
                        "初始 0, 按 V 变 1, 再按回 0; 帧对齐, 用于锁定 value=1 的帧")
    p.add_argument("--show-camera", action="store_true",
                   help="实时显示摄像头画面 (人工遥控/录制观察用)")
    p.add_argument("--start-teleop", action="store_true",
                   help="启动即进入人工遥控模式 (默认推理)。按 h 随时切换 遥控↔推理; "
                        "遥控段同样录进数据集, 适合遥控引导录制")
    p.add_argument("--joint-limits-file", default="",
                   help="数据集关节绝对限幅 JSON (extract_joint_limits.py 生成)。"
                        "所有执行动作 (含推理/返回段) 都会被钳制到数据集范围内。空=不加载")

    # 注意: parse_args 需要去掉 argv[0] (脚本路径), 否则会被当成未知参数
    args = p.parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [REC] %(message)s", force=True)
    log = logging.getLogger("record_ws")

    num_steps = max(1, min(args.num_action_steps, 30))

    # 发送给 server 的力全 0 消融: --no-force 或 --zero-force-to-server 任一生效
    # force_buf 始终不填充 → build_wrench_history 返回全 0 [60,6] 发给模型
    # 数据集 observation.force 仍记录本地真实力 (不依赖此开关)
    send_zero_force = bool(args.no_force or args.zero_force_to_server)
    if send_zero_force:
        log.info(">>> 力消融: 发送给 server 的力全 0 (数据集仍记录本地真实力)")

    # --no-server 纯本地手控: 必须 --start-teleop (全程遥控, 无推理能力)
    if args.no_server and not args.start_teleop:
        log.error("--no-server 只能配合 --start-teleop (纯遥控录制, 无推理)")
        sys.exit(1)

    # 加载数据集关节绝对限幅
    global JOINT_LIMITS
    if args.joint_limits_file:
        import json as _json
        with open(args.joint_limits_file, "r", encoding="utf-8") as _f:
            _jl = _json.load(_f)
        _IDX = {"j1": 0, "j2": 1, "j3": 2, "j4": 3, "j5": 4, "j6": 5, "gripper": 6}
        JOINT_LIMITS = {}
        for _name, _lim in _jl.get("joints", {}).items():
            _i = _IDX.get(_name)
            if _i is not None:
                JOINT_LIMITS[_i] = (float(_lim["min"]), float(_lim["max"]))
        log.info("已加载关节限幅: %s", args.joint_limits_file)
        log.info("  j1=[%+.3f,%+.3f] j2=[%+.3f,%+.3f] j3=[%+.3f,%+.3f] j4=[%+.3f,%+.3f] j5=[%+.3f,%+.3f] j6=[%+.3f,%+.3f] gripper=[%+.3f,%+.3f]",
                 *[v for i in range(7) for v in JOINT_LIMITS.get(i, (0.0, 0.0))])

    # ── 1. 连接机器人 ──
    log.info("Connecting robot...")
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()

    # ── 2. 创建/加载数据集 (自动包含 observation.force 当前帧特征) ──
    if args.resume:
        dataset = LeRobotDataset(args.repo_id)
        if len(robot.cameras) > 0:
            dataset.start_image_writer()
    else:
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            robot=robot,
            use_videos=args.video,
        )
    log.info("Dataset features: %s", list(dataset.features))

    # 可选: 记录模型预测力 force_pred → 手动加 feature (meta.info 在内存, 改后写回)
    # 注意: shape 必须用 tuple (与 robot.force_features 一致), 否则 validate_frame 严格比较
    #       actual_shape(tuple) != expected_shape(list) 会报错
    if args.record_force_pred:
        dataset.meta.info["features"]["observation.force_pred"] = {
            "dtype": "float32",
            "shape": (FORCE_DIM,),
            "names": ["Fx_pred", "Fy_pred", "Fz_pred", "Tx_pred", "Ty_pred", "Tz_pred"],
        }
        import json as _json
        _info_path = Path(dataset.root) / "meta" / "info.json"
        with open(_info_path, "w", encoding="utf-8") as _f:
            _json.dump(dataset.meta.info, _f, ensure_ascii=False, indent=2)
        # 重新初始化 episode_buffer: create 时已按旧 features (无 force_pred) 建好,
        # 若不复建, add_frame 会 KeyError: 'observation.force_pred'
        dataset.episode_buffer = dataset.create_episode_buffer()
        # 重建 hf_dataset: _save_episode_table 用 self.hf_features 取列, 而 hf_dataset 是
        # create 时(无 force_pred)建的, 不重建则 force_pred 列会被丢弃 (parquet 无该列)
        dataset.hf_dataset = dataset.create_hf_dataset()
        log.info("已启用 force_pred 记录: observation.force_pred [%d] (hf_dataset 已重建)", FORCE_DIM)

    if args.record_value:
        dataset.meta.info["features"]["observation.value"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["value"],
        }
        import json as _json
        _info_path = Path(dataset.root) / "meta" / "info.json"
        with open(_info_path, "w", encoding="utf-8") as _f:
            _json.dump(dataset.meta.info, _f, ensure_ascii=False, indent=2)
        # 同 force_pred: 重建 episode_buffer + hf_dataset, 否则 add_frame KeyError / parquet 缺列
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.hf_dataset = dataset.create_hf_dataset()
        log.info("已启用 value 记录: observation.value (int64, V键切换 0/1) (hf_dataset 已重建)")

    # ── 3. 连接 server (--no-server 纯本地手控时跳过) ──
    ws = None
    packer = None
    if not args.no_server:
        log.info("Connecting server %s ...", args.server)
        ws = websockets.sync.client.connect(
            args.server, compression=None, max_size=None,
            ping_interval=None, ping_timeout=None,
        )
        metadata = msgpack_numpy.unpackb(ws.recv())
        log.info("Server metadata: %s", metadata)
        packer = msgpack_numpy.Packer()
    else:
        log.info("--no-server: 纯本地手控录制 (不连接 server; 需全程遥控, 按 h 切推理会被拒绝)")

    try:
        for ep in range(args.num_episodes):
            ep_attempt = 0
            # ← 重置轨迹时: 丢弃当前数据, 重新录制同一 episode (attempt +1)
            while True:
                ep_attempt += 1
                log.info("=== Episode %d (attempt %d) ===", ep, ep_attempt)
                force_buf = deque(maxlen=FT_HISTORY_STEPS)
                action_chunk = None
                chunk_pos = 0
                value = 0                              # V 键切换 0/1 (逐帧记录, 每 episode 重置)
                teleop_mode = bool(args.start_teleop)   # h 切换: False=模型推理, True=人工遥控
                quit_all = False               # ESC: 完成当前轨迹并退出整个录制
                reset_requested = False        # ←: 丢弃重录
                result_label = None            # S/F/→: None=无标注, success/fail

                # (a0) 归零: 先回到启动零位 (arm.hold_position, 不用推理数值)
                #      稳定后再开始 warmup 推理 → 起点是真实静止位置, 不漂移
                move_to_home(robot, hold_s=1.5)

                # (b) 录制起点 = 归零后的实际位姿 (作为结束段返回目标 / ←重置后回到这里)
                start_obs = robot.capture_observation()
                start_state = np.asarray(start_obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM].tolist()
                log.info("录制起点: %s", [f"{v:+.3f}" for v in start_state])

                if not teleop_mode:
                    # (a) warmup: 从归零后的起点开始推理, 让模型进入状态 + 攒力历史
                    tw = time.perf_counter()
                    while time.perf_counter() - tw < args.warmup_time_s:
                        obs = robot.capture_observation()
                        if args.show_camera:
                            show_cameras(obs, robot.cameras)
                        f = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                        if not send_zero_force:
                            force_buf.append(f)
                        # warmup 期间也推理执行, 让模型进入状态
                        if action_chunk is None or chunk_pos >= num_steps:
                            if ws is None:
                                log.error("--no-server 模式无推理能力 (warmup 跳过)")
                                break
                            payload = build_observation(obs, force_buf, args.task)
                            ws.send(packer.pack(payload))
                            raw = ws.recv()
                            resp = msgpack_numpy.unpackb(raw) if not isinstance(raw, str) else None
                            if resp is None:
                                log.error("Warmup server error: %s", raw)
                                break
                            action_chunk = np.asarray(resp["actions"])
                            chunk_pos = 0
                        target = action_chunk[min(chunk_pos, len(action_chunk) - 1)].tolist()
                        chunk_pos += 1
                        cur = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
                        safe = clamp_action_delta(target, cur.tolist(),
                                                  args.max_joint_delta, args.max_gripper_delta)
                        robot.send_action(torch.tensor(safe, dtype=torch.float32))
                        busy_wait(1.0 / args.fps)
                    log.info("Warmup done (%d force frames)", len(force_buf))
                else:
                    log.info(">>> 启动模式: 人工遥控 (按 h 切换推理) — 遥控段直接录制")

                # (b) 录制 episode
                step = 0
                te = time.perf_counter()
                done = False
                while time.perf_counter() - te < args.episode_time_s and not done:
                    t0 = time.perf_counter()

                    # 键盘监听: S成功 | F失败 | →无标注 | ←重置 | h遥控/推理切换 | ESC完成并退出
                    for ev in poll_keys():
                        if ev == "done_success":
                            log.info("S pressed — 标注【成功】并结束轨迹")
                            result_label = "success"
                            done = True
                        elif ev == "done_fail":
                            log.info("F pressed — 标注【失败】并结束轨迹")
                            result_label = "fail"
                            done = True
                        elif ev == "done_none":
                            log.info("→ pressed — 无标注结束轨迹 (none)")
                            result_label = None
                            done = True
                        elif ev == "reset":
                            log.info("← pressed — 重置轨迹 (丢弃当前帧, 回到起始重新录制)")
                            reset_requested = True
                            done = True
                        elif ev == "toggle":
                            teleop_mode = not teleop_mode
                            if teleop_mode:
                                log.info("h pressed — 切换到【人工遥控】(再按 h 回模型推理)")
                            else:
                                log.info("h pressed — 切换到【模型推理】")
                                action_chunk = None      # 回推理时重新查询
                                chunk_pos = 0
                        elif ev == "value":
                            value = 1 - value
                            log.info("V pressed — value 标注 → %d", value)
                        elif ev == "quit":
                            log.info("ESC pressed — 完成当前轨迹并退出录制")
                            done = True
                            quit_all = True
                    if done:
                        break

                    if teleop_mode:
                        # 人工遥控: teleop_step 直接执行 leader 动作, 返回 obs+action 录帧
                        obs_t, act_t = robot.teleop_step(record_data=True)
                        if args.show_camera:
                            show_cameras(obs_t, robot.cameras)
                        f_now = np.asarray(obs_t[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                        if not send_zero_force:
                            force_buf.append(f_now)      # 遥控期间也攒力历史
                        frame = {"task": args.task}
                        for k, v in obs_t.items():
                            frame[k] = v
                        frame["action"] = act_t["action"]
                        if args.record_force_pred:
                            frame["observation.force_pred"] = torch.zeros(
                                FORCE_DIM, dtype=torch.float32)   # 遥控段无预测, 补零
                        if args.record_value:
                            frame["observation.value"] = torch.as_tensor([value], dtype=torch.int64)
                        dataset.add_frame(frame)
                        step += 1
                        dt = time.perf_counter() - t0
                        if dt < 1.0 / args.fps:
                            busy_wait(1.0 / args.fps - dt)
                        continue

                    obs = robot.capture_observation()
                    if args.show_camera:
                        show_cameras(obs, robot.cameras)

                    # 当前帧力 → 历史 buffer (发送给模型; send_zero_force 时不填充 → 全0)
                    f_now = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                    if not send_zero_force:
                        force_buf.append(f_now)

                    # 推理 (chunk 耗尽时); --no-server 无推理能力 → 强制回遥控
                    if action_chunk is None or chunk_pos >= num_steps:
                        if ws is None:
                            log.error("--no-server 模式不能切推理 (按 h 仅遥控)")
                            teleop_mode = True
                            action_chunk = None
                            chunk_pos = 0
                            continue
                        payload = build_observation(obs, force_buf, args.task)
                        ws.send(packer.pack(payload))
                        raw = ws.recv()
                        if isinstance(raw, str):
                            log.error("Server error: %s", raw)
                            done = True
                            break
                        resp = msgpack_numpy.unpackb(raw)
                        action_chunk = np.asarray(resp["actions"])
                        fp_chunk = None
                        if args.record_force_pred and "force_pred" in resp:
                            fp_chunk = np.asarray(resp["force_pred"], dtype=np.float32)   # (30,6)
                        chunk_pos = 0

                    # 取目标 + 限幅 + 执行
                    target = action_chunk[min(chunk_pos, len(action_chunk) - 1)].tolist()
                    chunk_pos += 1
                    cur_state = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
                    safe = clamp_action_delta(target, cur_state.tolist(),
                                              args.max_joint_delta, args.max_gripper_delta)
                    robot.send_action(torch.tensor(safe, dtype=torch.float32))

                    # ── 录帧: state(7) + force(当前帧6) + action(7) + images (+ force_pred) ──
                    frame = {
                        "observation.state": torch.as_tensor(cur_state, dtype=torch.float32),
                        "observation.force": torch.as_tensor(f_now, dtype=torch.float32),  # 当前帧
                        "action": torch.as_tensor(safe, dtype=torch.float32),
                        "task": args.task,
                    }
                    if args.record_force_pred:
                        if fp_chunk is not None:
                            fp = fp_chunk[min(chunk_pos - 1, len(fp_chunk) - 1)]   # 当前步对应行
                        else:
                            fp = np.zeros(FORCE_DIM, dtype=np.float32)              # server 未返回, 补零
                        frame["observation.force_pred"] = torch.as_tensor(fp, dtype=torch.float32)
                    if args.record_value:
                        frame["observation.value"] = torch.as_tensor([value], dtype=torch.int64)
                    for name in robot.cameras:
                        img = obs[f"observation.images.{name}"]
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        frame[f"observation.images.{name}"] = img
                    dataset.add_frame(frame)
                    step += 1

                    # 帧率控制
                    dt = time.perf_counter() - t0
                    if dt < 1.0 / args.fps:
                        busy_wait(1.0 / args.fps - dt)

                # (b1) ← 重置轨迹: 丢弃当前数据, 回到起始位置, 重新录制
                if reset_requested:
                    log.info("← 丢弃 episode %d 数据 (%d 帧), 回到起始位置...", ep, step)
                    dataset.episode_buffer = None        # 丢弃内存 buffer (不 save)
                    replay_to_start(robot, start_state, args.fps)   # 低速归位
                    continue                             # 重新 warmup + 录制同一 episode

                # (b2) 结束段: 松开/返回初始位置 (录进数据集, 形成完整闭环)
                log.info("Returning to start pose...")
                replay_to_start(robot, start_state, args.fps)   # 低速归位
                # 录制返回动作 (取当前位姿执行直到接近起始位)
                # 注意: 用 arm.write 而非 send_action —— send_action 走 policy_joint_delta_limit
                # 限幅 (0.025/步) 会让归位又慢又卡; arm.write 与 replay_to_start 一致, 彻底绕过
                # 返回段同样低速小步进, 减小电机负载/发热
                _old_spd = _lower_speed(robot)
                tr_end = time.perf_counter()
                try:
                    while time.perf_counter() - tr_end < 8.0:  # 最多 8 秒返回段 (低速)
                        obs = robot.capture_observation()
                        f_now = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                        cur_state = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
                        # 目标 = 起始位置, 限幅平滑
                        safe = clamp_action_delta(start_state, cur_state.tolist(),
                                                  RETURN_JOINT_DELTA, RETURN_GRIPPER_DELTA)
                        robot.arm.write(safe)                # 绕过 policy 限幅, 直接写
                        frame = {
                            "observation.state": torch.as_tensor(cur_state, dtype=torch.float32),
                            "observation.force": torch.as_tensor(f_now, dtype=torch.float32),
                            "action": torch.as_tensor(safe, dtype=torch.float32),
                            "task": args.task,
                        }
                        if args.record_force_pred:
                            frame["observation.force_pred"] = torch.zeros(
                                FORCE_DIM, dtype=torch.float32)   # 返回段无预测, 补零
                        if args.record_value:
                            frame["observation.value"] = torch.as_tensor([value], dtype=torch.int64)
                        for name in robot.cameras:
                            img = obs[f"observation.images.{name}"]
                            if isinstance(img, torch.Tensor):
                                img = img.cpu().numpy()
                            frame[f"observation.images.{name}"] = img
                        dataset.add_frame(frame)
                        step += 1
                        # 已接近起始位置 (容差内) → 停止返回段
                        if np.abs(cur_state - np.array(start_state[:PROPRIO_DIM])).max() < RETURN_TOL:
                            break
                        busy_wait(1.0 / args.fps)
                finally:
                    robot.arm.set_move_speed(_old_spd)

                dataset.save_episode()
                # 真实保存的 episode_index: save_episode 后 meta.total_episodes 已 +1
                # (resume 时从已有条数累加; 主循环 ep 每次都从 0 起, 不能用作标注)
                ep_index = dataset.meta.total_episodes - 1
                log.info("Episode %d saved (real index %d, %d frames), 结果标注=%s",
                         ep, ep_index, step, result_label)
                # 保存结果标注 (success/fail/none) 到 results.jsonl
                _results_path = Path(dataset.root) / "meta" / "results.jsonl"
                _results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_results_path, "a", encoding="utf-8") as _rf:
                    _rf.write(json.dumps({"episode_index": ep_index, "result": result_label,
                                          "frames": step, "task": args.task},
                                         ensure_ascii=False) + "\n")
                break                                 # 当前 episode 完成

            # (c) reset 环境 (不录制, 完全关闭推理: 机械臂停在返回段结束位置等待)
            #     reset 段不做任何推理/send_action —— 下一 episode 会重新归零再开始
            if quit_all:
                log.info("ESC — 停止后续 episode, 退出录制")
                break
            if ep < args.num_episodes - 1:
                log.info("Reset 等待 %.1fs (推理已关闭, 机械臂停在返回段结束位置)...",
                         args.reset_time_s)
                tr = time.perf_counter()
                # 完全关闭推理: 只保持当前位置 (arm.write 当前值, 不 send_action 不查推理)
                while time.perf_counter() - tr < args.reset_time_s:
                    obs = robot.capture_observation()
                    cur = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
                    robot.arm.write(cur.tolist())    # 保持当前位置, 无推理
                    busy_wait(1.0 / args.fps)
                # 彻底清空推理状态, 确保下一 episode 从遥控/归零干净开始
                action_chunk = None
                chunk_pos = 0
                log.info("Reset done")

        log.info("Dataset recording complete: %s (%d episodes)",
                 args.repo_id, dataset.num_episodes)

    finally:
        log.info("Disconnecting...")
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        log.info("Done")


if __name__ == "__main__":
    main()
