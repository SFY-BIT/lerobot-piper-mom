#!/usr/bin/env python

import argparse
import logging
from typing import Any

import torch
import zmq

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def summarize_normalization_buffers(policy: PI0Policy) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    modules = {
        "normalize_inputs": policy.normalize_inputs,
        "normalize_targets": policy.normalize_targets,
        "unnormalize_outputs": policy.unnormalize_outputs,
    }

    for module_name, module in modules.items():
        bad_keys: list[str] = []
        for key in module.features:
            buffer_name = "buffer_" + key.replace(".", "_")
            if not hasattr(module, buffer_name):
                continue
            buffer = getattr(module, buffer_name)
            for stat_name, value in buffer.items():
                if torch.isinf(value).any():
                    bad_keys.append(f"{key}.{stat_name}")
        if bad_keys:
            missing[module_name] = bad_keys
    return missing


def build_batch(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "observation.state": torch.tensor(payload["state"], dtype=torch.float32, device=device).unsqueeze(0),
        "task": [payload["task"]],
    }

    for key, image in payload["images"].items():
        tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
        batch[key] = tensor.unsqueeze(0)

    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a remote pi0 inference server.")
    parser.add_argument("--policy-path", required=True, help="Local path to the pi0 pretrained_model directory.")
    parser.add_argument("--device", default="cuda", help="Torch device to run inference on.")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555", help="ZeroMQ bind address.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device)
    logging.info("Loading pi0 policy from %s on %s", args.policy_path, device)
    policy = PI0Policy.from_pretrained(args.policy_path, map_location=str(device))
    policy.to(device)
    policy.eval()
    policy.reset()

    missing_stats = summarize_normalization_buffers(policy)
    if missing_stats:
        for module_name, keys in missing_stats.items():
            logging.error("Missing normalization stats in %s: %s", module_name, ", ".join(keys))
        raise RuntimeError(
            "Loaded pi0 policy is missing normalization statistics. "
            "This export cannot be used for direct LeRobot inference."
        )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    logging.info("Listening on %s", args.bind)

    try:
        while True:
            message = socket.recv_pyobj()
            msg_type = message.get("type")

            if msg_type == "reset":
                policy.reset()
                socket.send_pyobj({"status": "ok"})
                continue

            if msg_type != "infer":
                socket.send_pyobj({"status": "error", "error": f"Unsupported message type: {msg_type}"})
                continue

            try:
                batch = build_batch(message, device)
                with torch.inference_mode():
                    action = policy.select_action(batch)
                socket.send_pyobj({"status": "ok", "action": action.squeeze(0).detach().cpu().tolist()})
            except Exception as exc:  # noqa: BLE001
                logging.exception("Inference failed")
                socket.send_pyobj({"status": "error", "error": repr(exc)})
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
