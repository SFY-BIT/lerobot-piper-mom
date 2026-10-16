#!/usr/bin/env python3
"""Piper 真机推理客户端 — websocket 版（适配 60 帧力历史 K=16 模型）

运行位置: 真机端 (或任何能采集真机 obs 的机器)
Server 位置: 训练/推理机 (serve_policy.py, websocket 端口 8000)

┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ 关键设计: 60 帧力历史由 CLIENT 端拼接, server 无状态不拼        │
│                                                                     │
│  Server 的 ForceInStatePiperInputs 期望收到完整 wrench_history:     │
│    observation/wrench_history: [60, 6]  (60 帧力/力矩历史)          │
│  如果 client 不发, server 会用"当前帧力重复 60 次"兜底 ——          │
│  那会丢失真实历史, 本 client 负责真正攒 60 帧。                     │
│                                                                     │
│  不足 60 帧时: 前面补零 (与数据集 stamp_seal_v2_flexiv_ft60 一致):  │
│    数据集 wrench_history 语义 = 当前帧往前 60 帧 (含当前),           │
│    轨迹开头不足 60 帧 → 最早的帧前面补 0。                          │
└─────────────────────────────────────────────────────────────────────┘

发送给 server 的观测 (与 offline_replay 完全一致):
  observation/image          : 主相机 RGB (H,W,3) uint8, server 会 resize 224
  observation/wrist_image    : 腕相机 RGB (H,W,3) uint8
  observation/state          : 7 维 [j1..j6, gripper]  (proprio, 单位 rad)
  observation/wrench_history : [60, 6] float32
                              列序 = [Fx,Fy,Fz,Tx,Ty,Tz] (与数据集一致)
  prompt                     : 任务文本, 如 "stamp seal"

接收 (server → client):
  actions    : (30, 7) 绝对关节位置 [j1..j6, gripper] rad
  force_pred : (30, 6) 预测力 (仅 predict_force 模型, 可选)

执行语义 (重要):
  - 模型内部输出的是 DELTA (动作增量, DeltaActions 输入侧做 delta 精度控制)
  - 但 server 的 output_transform 链里 AbsoluteActions 会把 delta 加回 state:
        actions_abs = actions_delta + state   (mask 前6关节)
    因此 client 收到的是"绝对目标关节位置" (30,7)
  - send_action 传绝对目标即可 (勿再转 delta, 否则双重转换出错)
  - 6×30 force_pred 仅预测力, 不用于控制
  - 每帧执行前做限幅 (相对当前关节, 防猛动)

频率控制 (解耦 执行/推理):
  --fps             执行频率 (send_action 调用频率, Hz)
  --num-action-steps 每次推理后连续执行的 chunk 步数 (1~30)
                     → 推理频率 = fps / num_action_steps
                     例: --fps 30 --num-action-steps 5 → 执行 30Hz, 推理 6Hz
  --infer-fps       可选推理频率上限 (留作未来限流, 当前由 chunk 步数决定)

用法 (真机端):
  # 最响应 (每步推理): 
  python piper_ws_force_client.py --server ws://<IP>:8000 --task "stamp seal" \
      --fps 30 --num-action-steps 1 --duration-s 120
  # 省算力 (推理 6Hz):
  python piper_ws_force_client.py --server ws://<IP>:8000 --task "stamp seal" \
      --fps 30 --num-action-steps 5 --duration-s 120

依赖 (真机端):
  websockets  msgpack_numpy  numpy  torch  lerobot(含 Piper 机器人)
"""

import argparse
import base64
import json
import logging
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── 非阻塞键盘监听 (Windows: msvcrt) ──
try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

# 按键码
_KEY_LEFT = b"\xe0K"      # ←
_KEY_RIGHT = b"\xe0M"     # →
_KEY_UP = b"\xe0H"        # ↑
_KEY_DOWN = b"\xe0P"      # ↓
_KEY_ESC = b"\x1b"
_KEY_F = b"f"             # F: 切换 遥控/推理
_KEY_V = b"v"             # V: 切换 value 标注 0/1 (兼容大写)
_KEY_SPACE = b" "


def poll_keys() -> list[str]:
    """非阻塞读取按键, 返回按下的事件列表 (如 ['right', 'esc', 'f'])。"""
    events = []
    if not _HAS_MSVCRT:
        return events
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):        # 方向键/功能键双字节
            ch2 = msvcrt.getch()
            ch = ch + ch2
        if ch == _KEY_RIGHT:
            events.append("right")
        elif ch == _KEY_LEFT:
            events.append("left")
        elif ch == _KEY_ESC:
            events.append("esc")
        elif ch == _KEY_F:
            events.append("toggle")
        elif ch in (_KEY_V, b"V"):
            events.append("value")
        elif ch == _KEY_SPACE:
            events.append("space")
    return events

# ── openpi-client 的 websocket + msgpack 编码 ──
# 仓库布局 (src-layout): packages/pyproject.toml + packages/src/openpi_client/
# 向上逐级查找 packages/src/openpi_client, 兼容脚本放不同目录
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

# ── 真机机器人 (复用旧版 piper_force_client 的 lerobot 采集) ──
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait

# =========================================================================
# 配置常量 — 必须与 server 端训练配置一致
# =========================================================================
FT_HISTORY_STEPS = 60      # 力历史帧数 T (训练配置 ft_history_steps=60)
FORCE_DIM = 6              # 力/力矩维数 (Fx,Fy,Fz,Tx,Ty,Tz)
PROPRIO_DIM = 7            # 关节+夹爪 (force_start_idx=7)
ACTION_DIM = 7             # 模型输出关节动作维 (joints6 + gripper)
ACTION_HORIZON = 30        # server 返回的动作块长度

# 真机 lerobot obs 键 → server 期望键 (按你的真机环境调整!)
#   旧版 client 用 observation.images.one/two, 这里映射到 server 键
CAMERA_MAIN = "observation.images.one"      # 主相机 (本地键)
CAMERA_WRIST = "observation.images.two"     # 腕相机 (本地键)
STATE_KEY = "observation.state"             # 7 维关节+夹爪
FORCE_KEY = "observation.force"             # 6 维力/力矩 (真机自定义)


def build_wrench_history(force_buf: deque) -> np.ndarray:
    """把 60 帧力历史 deque 拼成 [60, 6] ndarray。

    与数据集处理一致:
      - deque 尾部 = 最新帧
      - 不足 60 帧 → 前面补零 (轨迹开头的历史未知, 用 0 填充)
    """
    frames = list(force_buf)                    # 旧→新 (最旧在前)
    T_avail = len(frames)
    if T_avail >= FT_HISTORY_STEPS:
        return np.stack(frames[-FT_HISTORY_STEPS:], axis=0)  # 只留最近 60
    # 补零到 60: [pad_zeros, ...真实帧] → 最早的帧补 0
    pad = np.zeros((FT_HISTORY_STEPS - T_avail, FORCE_DIM), dtype=np.float32)
    real = np.stack(frames, axis=0)
    return np.concatenate([pad, real], axis=0)  # [60, 6]


def build_observation(obs: dict[str, Any], force_buf: deque,
                      prompt: str,
                      camera_main: str = CAMERA_MAIN,
                      camera_wrist: str = CAMERA_WRIST,
                      gripper_scale: float = 1.0) -> dict[str, Any]:
    """把真机 obs + 力 buffer 组装成 server 期望的 payload。

    gripper_scale: 夹爪尺度缩放 (默认 1.0)
      若 >1 (如 1.1): 发给模型的夹爪 state = 实际 / 1.1
      (配合执行时 target*1.1, 模型看到的数值不变, 物理执行放大)
    """
    # 相机: 转成 uint8 ndarray (server 端会 resize 到 224)
    img = _to_uint8(obs[camera_main])
    wrist = _to_uint8(obs[camera_wrist])

    # state: 7 维 proprio (只取前 7, 若真机返回 13 维含力则截断)
    state = np.asarray(obs[STATE_KEY], dtype=np.float32)
    if state.ndim > 1:
        state = state[0]                      # 兼容 (1,7) 形状
    proprio = state[:PROPRIO_DIM].copy()
    if gripper_scale > 0:
        proprio[-1] = proprio[-1] / gripper_scale   # 夹爪反馈 ÷ scale

    # wrench_history: client 端拼接的 60 帧历史 [60,6]
    wrench_history = build_wrench_history(force_buf)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": proprio.tolist(),
        "observation/wrench_history": wrench_history,   # [60,6] ← client 拼!
        "prompt": prompt,
    }


def _to_uint8(img: Any) -> np.ndarray:
    """把 torch.Tensor / ndarray 统一成 uint8 RGB ndarray。"""
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def img_to_jpeg_b64(img: Any, scale: float = 0.5, quality: int = 80) -> str | None:
    """把图像转成 JPEG base64 字符串 (用于嵌入日志)。

    scale: 缩放比例 (0.5 = 320x240, 控制日志体积)
    quality: JPEG 质量 (80 默认)
    返回: "data:image/jpeg;base64,..." 或 None (失败)
    """
    if not _HAS_CV2:
        return None
    try:
        arr = _to_uint8(img)
        if scale != 1.0:
            h, w = arr.shape[:2]
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return b64
    except Exception:
        return None


def show_cameras(obs: dict[str, Any], camera_main: str, camera_wrist: str) -> None:
    """执行时实时显示两个摄像头画面 (主相机 + 腕相机)。"""
    if not _HAS_CV2:
        return
    try:
        main = _to_uint8(obs.get(camera_main))
        wrist = _to_uint8(obs.get(camera_wrist))
        if main is not None:
            cv2.imshow("CAMERA 0 (main)", cv2.cvtColor(main, cv2.COLOR_RGB2BGR))
        if wrist is not None:
            cv2.imshow("CAMERA 1 (wrist)", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except Exception as e:
        logging.warning("show_cameras failed: %s", e)


def clamp_action_delta(target: list[float], current: list[float],
                       max_joint_delta: float, max_gripper_delta: float,
                       ) -> tuple[list[float], bool]:
    """限幅动作: 防止单步关节/夹爪变化过大 (安全)。"""
    clamped = list(target)
    was_clamped = False
    for i, (t, c) in enumerate(zip(target, current)):
        max_d = max_gripper_delta if i == len(target) - 1 else max_joint_delta
        lo, hi = c - max_d, c + max_d
        bounded = max(lo, min(hi, float(t)))
        if bounded != float(t):
            was_clamped = True
        clamped[i] = bounded
    return clamped, was_clamped


def main() -> None:
    p = argparse.ArgumentParser(description="Piper websocket force client (60-frame history)")
    p.add_argument("--server", default="ws://127.0.0.1:8000",
                   help="server 地址: ws://<IP>:8000 (serve_policy.py 端口)")
    p.add_argument("--task", default="stamp seal", help="任务指令 (prompt)")
    p.add_argument("--fps", type=int, default=30, help="执行频率 (send_action 频率, Hz)")
    p.add_argument("--num-action-steps", type=int, default=1,
                   help="每次推理后连续执行的 chunk 步数 (1=每步都推理最响应; "
                        "30=一次推理执行完整 30 步最省算力)")
    p.add_argument("--infer-fps", type=float, default=0,
                   help="推理频率上限 (Hz, 0=不限制)。推理结束后距上次推理不足 1/infer_fps "
                        "时会等待(推理间空隙)再执行下次推理。"
                        "实际推理频率 = min(fps/num_action_steps, infer_fps)")
    p.add_argument("--duration-s", type=float, default=120.0, help="最大运行时长")
    p.add_argument("--robot-type", default="piper", help="lerobot 机器人类型")
    p.add_argument("--max-joint-delta", type=float, default=0.2, help="单步关节最大变化 (rad)")
    p.add_argument("--max-gripper-delta", type=float, default=0.02, help="单步夹爪最大变化")
    p.add_argument("--gripper-scale", type=float, default=1.0,
                   help="夹爪执行增益 (1.0=原样, 1.1=执行幅度放大10%)。"
                        "实现: 发给模型的夹爪state÷scale, 执行target×scale, "
                        "模型看到的数值不变, 物理执行放大")
    p.add_argument("--q4-max-pos", type=float, default=0.0,
                   help="关节4 (q4) 正向绝对位置上限 (度)。0=无限制。"
                        "只限制正值 (q4 > 上限时钳制到上限), 负值不限制。"
                        "例: --q4-max-pos 15 → q4 最大 15° (0.2618 rad)")
    p.add_argument("--camera-main", default=CAMERA_MAIN, help="主相机 obs 键")
    p.add_argument("--camera-wrist", default=CAMERA_WRIST, help="腕相机 obs 键")
    p.add_argument("--no-force-history", action="store_true",
                   help="不攒历史, 只发当前帧力 (测试用, server 会 tile 60 次)")
    p.add_argument("--show-camera", action="store_true",
                   help="执行时弹出两个摄像头实时画面 (主相机 + 腕相机)")
    p.add_argument("--log-dir", default="", help="数据日志目录 (JSONL, 不含图片; 空=不记录)")
    p.add_argument("--log-name", default="",
                   help="日志文件名前缀 (如 lora16tk / full16tk / token1), 便于区分不同模型; 空=client_log")
    p.add_argument("--no-mark", action="store_true",
                   help="禁用键盘标注功能 (默认按→键切换 mark 1/0 写入日志)")
    p.add_argument("--start-teleop", action="store_true",
                   help="启动时进入遥控模式 (默认推理模式)。按 F 键随时切换 遥控↔推理")
    p.add_argument("--duration-s-teleop", type=float, default=0,
                   help="热切换时遥控段最大时长 (0=不限, 按 F 切回推理)")
    p.add_argument("--save-images", action="store_true",
                   help="每帧把两个摄像头画面以 JPEG base64 嵌入日志 (日志会很大)")
    p.add_argument("--image-scale", type=float, default=0.5,
                   help="嵌入日志的图片缩放比例 (0.5=320x240, 控制体积)")
    p.add_argument("--image-quality", type=int, default=80,
                   help="嵌入日志的 JPEG 质量 (0-100)")
    p.add_argument("--joint-limits-file", default="",
                   help="关节绝对限幅 JSON (由 extract_joint_limits.py 生成)。"
                        "推理 target 会被钳制到数据集范围内。空=不加载")
    args = p.parse_args()

    camera_main, camera_wrist = args.camera_main, args.camera_wrist
    infer_period = 1.0 / args.infer_fps if args.infer_fps > 0 else 0  # 推理间隔 (s)

    # 加载关节限幅 (从数据集提取的 JSON)
    joint_limits = None
    if args.joint_limits_file:
        import json as _json
        with open(args.joint_limits_file, "r", encoding="utf-8") as _f:
            jl = _json.load(_f)
        joint_limits = {}
        for name, lim in jl.get("joints", {}).items():
            idx = {"j1": 0, "j2": 1, "j3": 2, "j4": 3, "j5": 4, "j6": 5, "gripper": 6}.get(name)
            if idx is not None:
                joint_limits[idx] = (float(lim["min"]), float(lim["max"]))
        logging.info("已加载关节限幅: %s", args.joint_limits_file)
        logging.info("  j1=[%+.3f,%+.3f] j2=[%+.3f,%+.3f] j3=[%+.3f,%+.3f] j4=[%+.3f,%+.3f] j5=[%+.3f,%+.3f] j6=[%+.3f,%+.3f] gripper=[%+.3f,%+.3f]",
                     *[v for idx in range(7) for v in joint_limits.get(idx, (0, 0))])

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [CLIENT] %(message)s", force=True)
    logging.info("Config: fps=%d num_action_steps=%d infer_fps=%s (infer period=%s)",
                 args.fps, args.num_action_steps,
                 args.infer_fps or "off", f"{infer_period*1000:.0f}ms" if infer_period else "off")

    # ── 1. 连接真机机器人 ──
    logging.info("Connecting to Piper robot...")
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()
    logging.info("Robot connected")

    # ── 2. 连接 server (websocket) ──
    logging.info("Connecting to server %s ...", args.server)
    ws = websockets.sync.client.connect(
        args.server, compression=None, max_size=None,
        ping_interval=None, ping_timeout=None,
    )
    metadata = msgpack_numpy.unpackb(ws.recv())
    logging.info("Server connected, metadata=%s", metadata)
    packer = msgpack_numpy.Packer()

    # ── 3. 60 帧力历史 buffer (client 端拼接) ──
    force_buf = deque(maxlen=FT_HISTORY_STEPS)

    # ── 4. 数据日志 (JSONL, 不含图片) ──
    log_f = None
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.log_name if args.log_name else "client_log"
        log_path = Path(args.log_dir) / f"{name}_{ts}.jsonl"
        log_f = open(log_path, "w", encoding="utf-8")
        logging.info("数据日志: %s", log_path)

    try:
        # 先采集一帧用于初始化 buffer (轨迹开头: 补零逻辑见 build_wrench_history)
        init_obs = robot.capture_observation()
        init_force = np.asarray(init_obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
        force_buf.append(init_force)
        logging.info("Initialized force buffer (1 frame); will pad zeros until %d frames",
                     FT_HISTORY_STEPS)

        start_t = time.perf_counter()
        step = 0
        action_chunk = None          # 当前缓存的 (30,7) 动作块
        chunk_pos = 0                # 已消费的 chunk 内步数
        n_infer = 0                  # 推理次数统计
        last_infer_t = None          # 上次推理时间 (用于 --infer-fps 限速)
        mark = 0                     # 倾斜标注 (按 → 切换 0/1)
        value = 0                    # value 标注 (按 V 切换 0/1)
        teleop_mode = bool(args.start_teleop)   # True=遥控, False=推理
        teleop_start = time.perf_counter()
        if teleop_mode:
            logging.info(">>> 启动模式: 遥控 (按 F 切换推理)")
        else:
            logging.info(">>> 启动模式: 推理 (按 F 切换遥控)")
        while time.perf_counter() - start_t < args.duration_s:
            t0 = time.perf_counter()

            # (a0) 键盘监听: → 倾斜标注, F 切换模式, ESC 退出
            if not args.no_mark or True:   # 始终监听 (需要 F 键切换)
                for ev in poll_keys():
                    if ev == "right":
                        mark = 1 - mark
                        logging.info(">>> MARK 倾斜标注 → %d", mark)
                    elif ev == "value":
                        value = 1 - value
                        logging.info(">>> VALUE 标注 → %d", value)
                    elif ev == "toggle":
                        teleop_mode = not teleop_mode
                        teleop_start = time.perf_counter()
                        if teleop_mode:
                            logging.info(">>> 切换 → 遥控模式 (按 F 回推理)")
                            action_chunk = None   # 回推理时重新查询
                            chunk_pos = 0
                        else:
                            logging.info(">>> 切换 → 推理模式 (按 F 回遥控)")
                    elif ev == "esc":
                        logging.info("ESC pressed — exiting")
                        start_t = 0.0   # 强制退出循环
                        break

            # 遥控段超时自动回推理
            if (teleop_mode and args.duration_s_teleop > 0
                    and time.perf_counter() - teleop_start > args.duration_s_teleop):
                logging.info(">>> 遥控段超时 → 自动回推理")
                teleop_mode = False
                action_chunk = None
                chunk_pos = 0

            # (a) 采集真机观测 (每执行步都采集, 用于 state 限幅 + 力历史)
            obs = robot.capture_observation()

            # (a2) 实时显示两个视角 (主相机 + 腕相机)
            if args.show_camera:
                show_cameras(obs, camera_main, camera_wrist)

            # (a3) 遥控模式: 直接执行 leader 动作, 跳过推理
            if teleop_mode:
                teleop_result = robot.teleop_step(record_data=False)
                # 记录执行帧日志 (state/force/mark, 无 action/force_pred)
                if log_f is not None:
                    f_now = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM].tolist()
                    cur_st = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM].tolist()
                    entry = {
                        "t": round(time.perf_counter() - start_t, 4),
                        "step": step,
                        "query": False,
                        "chunk_id": -1,
                        "mode": "teleop",
                        "state": cur_st,
                        "force": f_now,
                        "mark": mark,
                        "value": value,
                    }
                    if args.save_images:
                        entry["image_main"] = img_to_jpeg_b64(
                            obs.get(camera_main), args.image_scale, args.image_quality)
                        entry["image_wrist"] = img_to_jpeg_b64(
                            obs.get(camera_wrist), args.image_scale, args.image_quality)
                    log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log_f.flush()
                step += 1
                dt = time.perf_counter() - t0
                if dt < 1.0 / args.fps:
                    busy_wait(1.0 / args.fps - dt)
                continue

            # (b) 把当前帧力 push 进历史 buffer (最新在末尾)
            if not args.no_force_history:
                force = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                force_buf.append(force)

            # (c) 需要推理时: chunk 耗尽 或 达到推理频率上限
            need_infer = (action_chunk is None) or (chunk_pos >= args.num_action_steps)
            if need_infer:
                # (c0) --infer-fps 限速: 距上次推理不足周期则等待 (推理间空隙)
                if infer_period > 0 and last_infer_t is not None:
                    gap = (last_infer_t + infer_period) - time.perf_counter()
                    if gap > 0:
                        time.sleep(gap)
                last_infer_t = time.perf_counter()

                payload = build_observation(obs, force_buf, args.task,
                                            camera_main, camera_wrist,
                                            args.gripper_scale)

                # (c1) 发送 → server 推理 → 接收
                tq = time.perf_counter()
                ws.send(packer.pack(payload))
                raw = ws.recv()
                qms = (time.perf_counter() - tq) * 1000
                if isinstance(raw, str):
                    logging.error("Server error: %s", raw)
                    break
                resp = msgpack_numpy.unpackb(raw)

                action_chunk = np.asarray(resp["actions"])       # (30,7) 绝对关节
                chunk_pos = 0
                n_infer += 1

                # 查询帧 (client 发出数据给 server): 单独标注 query=true
                if log_f is not None:
                    st_obs = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM].tolist()
                    f_now = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM].tolist()
                    fp_resp = resp.get("force_pred")
                    if fp_resp is not None:
                        fp_resp = np.asarray(fp_resp).tolist()
                    entry = {
                        "t": round(time.perf_counter() - start_t, 4),
                        "step": step,
                        "query": True,
                        "chunk_id": n_infer - 1,
                        "mode": "infer",
                        "rtt_ms": round(qms, 1),
                        "state": st_obs,
                        "force": f_now,
                        "force_sent": not args.no_force_history,
                        "force_pred": fp_resp,
                        "chunk_received": action_chunk.tolist(),
                        "wrench_history_steps": len(force_buf),
                        "mark": mark,          # 倾斜标注 (0/1)
                        "value": value,        # value 标注 (V键 0/1)
                    }
                    if args.save_images:
                        entry["image_main"] = img_to_jpeg_b64(
                            obs.get(camera_main), args.image_scale, args.image_quality)
                        entry["image_wrist"] = img_to_jpeg_b64(
                            obs.get(camera_wrist), args.image_scale, args.image_quality)
                    log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    log_f.flush()

            # (d) 取当前步目标 (绝对关节位置); 夹爪执行放大 ×scale
            target = action_chunk[min(chunk_pos, len(action_chunk) - 1)].tolist()
            chunk_pos += 1
            if args.gripper_scale != 1.0:
                target[-1] = target[-1] * args.gripper_scale   # 执行幅度放大
            # q4 正向绝对位置钳制 (只限制正值, 负值不动; 0=无限制)
            if args.q4_max_pos > 0:
                q4_lim = math.radians(args.q4_max_pos)
                if target[3] > q4_lim:
                    logging.warning("[step %d] q4 clamp: %+.4f → %+.4f (limit +%.1f°)",
                                    step, target[3], q4_lim, args.q4_max_pos)
                    target[3] = q4_lim
            # 数据集关节绝对限幅 (不超出数据集范围)
            if joint_limits is not None:
                for idx in range(len(target)):
                    if idx not in joint_limits:
                        continue
                    lo, hi = joint_limits[idx]
                    if target[idx] < lo or target[idx] > hi:
                        clamped_v = max(lo, min(hi, target[idx]))
                        logging.warning("[step %d] joint%d clamp: %+.4f → %+.4f (range [%+.4f, %+.4f])",
                                        step, idx + 1, target[idx], clamped_v, lo, hi)
                        target[idx] = clamped_v
            force_pred = np.asarray(resp["force_pred"])[min(chunk_pos - 1, len(action_chunk) - 1)].tolist() if "force_pred" in resp else None

            logging.info("[step %d] n_hist=%d infer#%d chunk[%d/%d] target=%s",
                         step, len(force_buf), n_infer, chunk_pos, args.num_action_steps,
                         [f"{v:+.4f}" for v in target])

            # (e) 安全限幅 (相对当前关节位置, 物理空间)
            current = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
            safe, was_clamped = clamp_action_delta(
                target, current.tolist(),
                args.max_joint_delta, args.max_gripper_delta,
            )
            # 数据集限幅也应用到最终执行值 safe (防止步进限幅后超出数据集范围)
            if joint_limits is not None:
                for idx in range(len(safe)):
                    if idx not in joint_limits:
                        continue
                    lo, hi = joint_limits[idx]
                    if safe[idx] < lo or safe[idx] > hi:
                        clamped_v = max(lo, min(hi, safe[idx]))
                        logging.warning("[step %d] joint%d safe-clamp: %+.4f → %+.4f (range [%+.4f, %+.4f])",
                                        step, idx + 1, safe[idx], clamped_v, lo, hi)
                        safe[idx] = clamped_v
            if was_clamped:
                logging.warning("[step %d] clamped: %s → %s",
                                step, [f"{v:+.4f}" for v in target],
                                [f"{v:+.4f}" for v in safe])

            # (f) 执行 (绝对关节目标, 必须是 torch.Tensor)
            robot.send_action(torch.tensor(safe, dtype=torch.float32))

            # 执行帧: 记录关节状态 + 实际发出指令 + 真实力 + 预测力
            if log_f is not None:
                entry = {
                    "t": round(time.perf_counter() - start_t, 4),
                    "step": step,
                    "query": False,
                    "chunk_id": n_infer - 1,
                    "mode": "infer",
                    "state": current.tolist(),
                    "action": safe,          # 发出的指令 (限幅后)
                    "target": safe,          # 别名, 兼容旧分析脚本
                    "force": np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM].tolist(),
                    "force_pred": force_pred, # 该帧对应的预测力
                    "mark": mark,            # 倾斜标注 (0/1)
                    "value": value,          # value 标注 (V键 0/1)
                }
                if args.save_images:
                    entry["image_main"] = img_to_jpeg_b64(
                        obs.get(camera_main), args.image_scale, args.image_quality)
                    entry["image_wrist"] = img_to_jpeg_b64(
                        obs.get(camera_wrist), args.image_scale, args.image_quality)
                log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_f.flush()

            # (g) 帧率控制
            dt = time.perf_counter() - t0
            logging.info("[step %d] dt=%.1fms fps=%.1f%s",
                         step, dt * 1000, 1.0 / dt if dt > 0 else 0,
                         f" force_pred={[f'{v:+.2f}' for v in force_pred]}" if force_pred else "")
            if args.fps:
                busy_wait(1.0 / args.fps - dt)
            step += 1

        # ── 会话统计 ──
        elapsed = time.perf_counter() - start_t
        logging.info("Session: %d steps in %.1fs (exec %.1f Hz), %d inferences "
                     "(infer %.1f Hz, chunk=%d)",
                     step, elapsed, step / elapsed if elapsed else 0,
                     n_infer, n_infer / elapsed if elapsed else 0,
                     args.num_action_steps)

    finally:
        logging.info("Disconnecting...")
        if log_f is not None:
            log_f.close()
            logging.info("数据日志已保存")
        if args.show_camera and _HAS_CV2:
            cv2.destroyAllWindows()
        try:
            ws.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        logging.info("Done")


if __name__ == "__main__":
    main()
