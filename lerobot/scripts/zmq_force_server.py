"""ZMQ 力感知推理服务器 —— Pi0Force 双头模型 (stamp seal)

接收 piper_force_client.py 的 ZMQ 请求，桥接为 policy.infer() 调用。

用法:
  cd /mnt/hdd/sfy/openpi-force
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python zmq_force_server.py \
      --checkpoint checkpoints/26000 --port 5555
"""

import argparse
import logging
import signal
import time

import jax
import numpy as np
import zmq

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
)
logger = logging.getLogger("zmq_force_server")

# ── 客户端图像 key → ForceInStatePiperInputs 期望的 key ──
# 客户端发送: images["observation.images.camera0"], images["observation.images.camera1"]
# ForceInStatePiperInputs 直接读取 (斜杠格式，无需 repack):
#   data["observation/image"] → base image
#   data["observation/wrist_image"] → wrist image
#   data["observation/state"] → state
CAMERA_KEY_MAP = {
    "observation.images.camera0": "observation/image",
    "observation.images.camera1": "observation/wrist_image",
}


def build_obs_from_request(request: dict) -> dict:
    """将客户端 ZMQ 请求转换为 policy.infer() 期望的 dict。

    客户端发送:
      {"state": [7], "force": [6], "images": {"observation.images.camera0": ..., ...}, "task": "..."}

    ForceInStatePiperInputs 期望 (斜杠格式，推理时不走 repack):
      {"observation/image": ndarray, "observation/wrist_image": ndarray,
       "observation/state": ndarray(13), "prompt": str}
    """
    state_7d = np.asarray(request["state"], dtype=np.float32)
    force_6d = np.asarray(request.get("force", [0.0] * 6), dtype=np.float32)
    state_13d = np.concatenate([state_7d, force_6d])  # [joints(7), force(6)]

    images = {}
    for client_key, policy_key in CAMERA_KEY_MAP.items():
        if client_key in request.get("images", {}):
            img = np.asarray(request["images"][client_key])
            images[policy_key] = img

    if not images:
        raise ValueError(
            f"No images found in request. Got keys: {list(request.get('images', {}))}"
        )

    obs = {
        "observation/state": state_13d,
        "prompt": request.get("task", "stamp seal"),
        **images,
    }
    return obs


def load_policy(checkpoint_dir: str, config_name: str):
    """加载 Pi0Force 策略。

    norm_stats 由 create_trained_policy 自动从 checkpoint_dir/assets/asset_id
    加载。由于 asset_id(=repo_id) 是绝对路径
    /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv，最终会从数据集目录读取。
    """
    logger.info("Loading config '%s'...", config_name)
    config = _config.get_config(config_name)

    logger.info("Creating policy from checkpoint '%s'...", checkpoint_dir)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    logger.info("Policy loaded successfully.")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Pi0Force ZMQ inference server")
    parser.add_argument(
        "--checkpoint", default="checkpoints/26000",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config", default="pi05_force_stamp_seal",
        help="Config name (pi05_force_stamp_seal or pi05_force_stamp_seal_remote)",
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="ZMQ server port",
    )
    args = parser.parse_args()

    # 加载策略 (首次推理触发 JIT 编译，需要几十秒)
    t0 = time.monotonic()
    policy = load_policy(args.checkpoint, args.config)
    logger.info("Load time: %.1fs", time.monotonic() - t0)

    # ZMQ 上下文
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_addr = f"tcp://*:{args.port}"
    socket.bind(bind_addr)
    logger.info("ZMQ server listening on %s", bind_addr)

    # 用 Poller 代替阻塞 recv，才能响应 Ctrl+C
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    # 优雅退出
    running = True

    def shutdown(signum, frame):
        nonlocal running
        logger.info("Received signal %d, shutting down...", signum)
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    step_count = 0
    try:
        while running:
            # 轮询等待（100ms 超时，可被 Ctrl+C 打断）
            socks = dict(poller.poll(timeout=100))
            if socket not in socks:
                continue

            try:
                request = socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                continue

            msg_type = request.get("type", "")

            if msg_type == "reset":
                socket.send_pyobj({"status": "ok"})
                logger.info("Policy reset")
                continue

            if msg_type != "infer":
                socket.send_pyobj({
                    "status": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
                continue

            # ── 推理 ──
            t_start = time.monotonic()
            try:
                obs = build_obs_from_request(request)
                result = policy.infer(obs)

                # 取完整 action chunk
                actions = np.asarray(result["actions"])
                action_chunk = actions.tolist()
                infer_ms = (time.monotonic() - t_start) * 1000

                reply = {
                    "status": "ok",
                    "action": action_chunk,
                    "infer_ms": infer_ms,
                }
                if "force_pred" in result:
                    force_pred = np.asarray(result["force_pred"])
                    reply["force_pred"] = force_pred.tolist()

                logger.info(
                    "step=%d chunk_shape=%s infer=%.0fms",
                    step_count,
                    list(actions.shape),
                    infer_ms,
                )
                step_count += 1

            except Exception as e:
                logger.error("Inference error: %s", e, exc_info=True)
                reply = {"status": "error", "message": str(e)}

            socket.send_pyobj(reply)

    finally:
        logger.info("Closing ZMQ socket...")
        socket.close(0)
        context.term()
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
