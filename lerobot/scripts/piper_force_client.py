"""力感知 Piper 推理客户端 (简洁同步版)

逻辑: 采集 → 推理 → 等速执行 → 重复

用法:
  python piper_force_client.py --task "stamp seal" --server tcp://10.96.45.13:5555
  python piper_force_client.py --task "stamp seal" --exec-steps 25 --interp-steps 0
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait

DEFAULT_CAMERA_MAPPING = {
    "observation.images.one": "observation.images.camera0",
    "observation.images.two": "observation.images.camera1",
}


def parse_camera_mappings(values):
    if not values:
        return dict(DEFAULT_CAMERA_MAPPING)
    m = {}
    for v in values:
        lk, rk = v.split("=", maxsplit=1)
        if not lk.startswith("observation.images."):
            lk = f"observation.images.{lk}"
        if not rk.startswith("observation.images."):
            rk = f"observation.images.{rk}"
        m[lk] = rk
    return m


def build_request(obs, task, cmap, force=True):
    state = obs["observation.state"].tolist()
    imgs = {}
    for lk, rk in cmap.items():
        if lk not in obs:
            continue
        x = obs[lk]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        imgs[rk] = x
    p: dict[str, Any] = {"type": "infer", "task": task, "state": state, "images": imgs}
    if force and "observation.force" in obs:
        f = obs["observation.force"]
        if isinstance(f, torch.Tensor):
            f = f.tolist()
        p["force"] = f
    return p


def interpolate_chunk(chunk, n_steps, n_interp):
    """取前 n_steps，相邻间插入 n_interp 帧"""
    n = min(n_steps, len(chunk))
    dim = len(chunk[0])
    r = []
    for i in range(n - 1):
        r.append(list(chunk[i]))
        for s in range(1, n_interp + 1):
            a = s / (n_interp + 1)
            r.append([chunk[i][d] * (1 - a) + chunk[i + 1][d] * a for d in range(dim)])
    r.append(list(chunk[n - 1]))
    return r


def clamp_action(target, current, max_joint, max_grip):
    c = list(target)
    for i in range(len(target)):
        mx = max_grip if i == len(target) - 1 else max_joint
        if mx is None or mx <= 0:
            continue
        lo, hi = current[i] - mx, current[i] + mx
        c[i] = max(lo, min(hi, float(target[i])))
    return c


def show_cameras(obs):
    """实时显示两个摄像头画面 (主相机 + 腕相机)。"""
    if not _HAS_CV2:
        return
    try:
        for lk, title in [("observation.images.one", "CAMERA 0 (main)"),
                          ("observation.images.two", "CAMERA 1 (wrist)")]:
            if lk not in obs:
                continue
            img = obs[lk]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            arr = img
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            cv2.imshow(title, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except Exception as e:
        logging.warning("show_cameras failed: %s", e)


def main():
    p = argparse.ArgumentParser(description="Piper sync client (v2 chunk-cache)")
    p.add_argument("--server", default="tcp://127.0.0.1:5555")
    p.add_argument("--task", required=True)
    p.add_argument("--exec-steps", type=int, default=25,
                   help="(兼容旧参数) 每次推理后执行的步数; 若未设 --num-action-steps 则等价于它")
    p.add_argument("--num-action-steps", type=int, default=0,
                   help="每次推理后执行的 chunk 步数 (1~30)。0=使用 --exec-steps。"
                        "推理频率 = fps / 该值")
    p.add_argument("--fps", type=float, default=30.0,
                   help="执行帧率 (Hz)。固定节奏执行, 每步 busy_wait 到 1/fps")
    p.add_argument("--interp-steps", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--robot-type", default="piper")
    p.add_argument("--camera-mapping", action="append")
    p.add_argument("--max-abs-joint-delta", type=float, default=0.2,
                   help="单步关节最大变化 (rad)。默认 0.2 (与新 client 一致, 更平滑)")
    p.add_argument("--max-abs-gripper-delta", type=float, default=0.02,
                   help="单步夹爪最大变化。默认 0.02")
    p.add_argument("--no-force", action="store_true")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--show-camera", action="store_true",
                   help="执行时弹出两个摄像头实时画面 (主相机 + 腕相机)")
    p.add_argument("--log-dir", default="", help="数据日志目录 (JSONL, 不含图片; 空=不记录)")
    p.add_argument("--log-name", default="",
                   help="日志文件名前缀 (如 lora16tk / full16tk / token1), 便于区分不同模型; 空=client_log")
    args = p.parse_args()

    # num_action_steps 优先, 未设置时用 exec-steps (兼容旧命令)
    if args.num_action_steps > 0:
        num_steps = args.num_action_steps
    else:
        num_steps = args.exec_steps
    num_steps = max(1, min(num_steps, 30))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [C] %(message)s")
    log = logging.getLogger("piper_client")
    cmap = parse_camera_mappings(args.camera_mapping)

    # 时序: 固定帧率执行
    frame_period = 1.0 / args.fps
    infer_hz = args.fps / num_steps
    log.info("num_steps=%d fps=%.0f → 推理 ~%.1fHz, frame_period=%.0fms",
             num_steps, args.fps, infer_hz, frame_period * 1000)

    # 连接
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.server)
    sock.send_pyobj({"type": "reset"})
    if sock.recv_pyobj().get("status") != "ok":
        raise RuntimeError("Reset failed")
    log.info("Connected")

    # 数据日志 (JSONL, 不含图片)
    log_f = None
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.log_name if args.log_name else "client_log"
        log_path = Path(args.log_dir) / f"{name}_{ts}.jsonl"
        log_f = open(log_path, "w", encoding="utf-8")
        log.info("数据日志: %s", log_path)

    step = 0
    ci = 0
    action_chunk = None          # 当前缓存的 (30,7) 动作块
    chunk_pos = 0                # 已消费的 chunk 内步数
    force_pred_all = None        # 当前 chunk 的预测力 (30,6)
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.duration_s:
            t_loop = time.perf_counter()

            # (a) 每步采集当前观测 (用于限幅 + 力 + 显示)
            obs = robot.capture_observation()
            if args.show_camera:
                show_cameras(obs)
            st = obs["observation.state"].tolist()

            # (b) chunk 耗尽时 → 查询推理 (阻塞)
            need_infer = (action_chunk is None) or (chunk_pos >= num_steps)
            if need_infer:
                req = build_request(obs, args.task, cmap, not args.no_force)
                tq = time.perf_counter()
                sock.send_pyobj(req)
                reply = sock.recv_pyobj()
                qms = (time.perf_counter() - tq) * 1000

                if reply.get("status") != "ok":
                    log.error("Server: %s", reply)
                    time.sleep(0.5)
                    continue

                action_chunk = reply["action"]
                chunk_pos = 0
                force_pred_all = reply.get("force_pred")   # (30,6) 或 None
                chunk_len = len(action_chunk)
                srv_ms = reply.get("infer_ms", 0)
                first3 = [[f"{v:+.4f}" for v in action_chunk[i]] for i in range(min(3, chunk_len))]
                last3 = [[f"{v:+.4f}" for v in action_chunk[i]] for i in range(max(0, chunk_len-3), chunk_len)]
                log.info("CHUNK %d: %d steps rtt=%.0fms srv=%.0fms | first=%s last=%s",
                         ci, chunk_len, qms, srv_ms, first3, last3)

                # 查询帧 (client 发出数据给 server): 单独标注 query=true
                if log_f is not None:
                    f_now = obs.get("observation.force")
                    if isinstance(f_now, torch.Tensor):
                        f_now = f_now.tolist()
                    log_f.write(json.dumps({
                        "t": round(time.perf_counter() - t0, 4),
                        "step": step,
                        "query": True,
                        "chunk_id": ci,
                        "rtt_ms": round(qms, 1),
                        "srv_ms": round(srv_ms, 1),
                        "state": st,
                        "force": f_now,
                        "force_sent": not args.no_force,
                        "force_pred": force_pred_all,
                        "chunk_received": action_chunk,
                    }, ensure_ascii=False) + "\n")
                    log_f.flush()
                ci += 1

            # (c) 取当前步目标 (绝对关节位置), 缓存复用
            target = list(action_chunk[min(chunk_pos, len(action_chunk) - 1)])
            chunk_pos += 1

            # (d) 安全限幅 (相对当前关节位置) — 默认开启, 与新 client 一致
            if args.no_clamp:
                safe = target
            else:
                safe = clamp_action(target, st, args.max_abs_joint_delta,
                                    args.max_abs_gripper_delta)

            # (e) 执行
            robot.send_action(torch.tensor(safe, dtype=torch.float32))
            step += 1

            # (f) 执行帧日志: state + action + force + force_pred
            if log_f is not None:
                f_now = obs.get("observation.force")
                if isinstance(f_now, torch.Tensor):
                    f_now = f_now.tolist()
                fp_this = None
                if force_pred_all is not None:
                    fp_this = force_pred_all[min(chunk_pos - 1, len(force_pred_all) - 1)]
                log_f.write(json.dumps({
                    "t": round(time.perf_counter() - t0, 4),
                    "step": step,
                    "query": False,
                    "chunk_id": ci - 1,
                    "state": st,
                    "action": safe,
                    "target": safe,
                    "force": f_now,
                    "force_pred": fp_this,
                }, ensure_ascii=False) + "\n")
                log_f.flush()

            if step % 50 == 0:
                fz = obs.get("observation.force", torch.zeros(6))
                if isinstance(fz, torch.Tensor):
                    fz = fz.tolist()
                log.info("step=%d fz=%+.1fN", step, fz[2] if isinstance(fz, list) else 0)

            # (g) 固定帧率
            elapsed = time.perf_counter() - t_loop
            if elapsed < frame_period:
                busy_wait(frame_period - elapsed)

    finally:
        log.info("Disconnecting (%d steps, %d chunks)", step, ci)
        if log_f is not None:
            log_f.close()
            log.info("数据日志已保存")
        if args.show_camera and _HAS_CV2:
            cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except Exception:
            pass
        sock.close(0)
        ctx.term()
        log.info("Done")


if __name__ == "__main__":
    main()
