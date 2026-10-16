"""
Teleoperation Agilex Piper with a PS5 controller
"""

import os
import time
import torch
import numpy as np
from dataclasses import replace

from lerobot.common.robot_devices.teleop.gamepad import create_piper_teleop_controller
from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig

# 力传感器（可选）
_force_cal = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
if _force_cal and os.path.isfile(_force_cal):
    try:
        from lerobot.common.robot_devices.sensors.force_sensor import ATIForceSensor
        _FORCE_AVAILABLE = True
    except ImportError:
        _FORCE_AVAILABLE = False
else:
    _FORCE_AVAILABLE = False


class PiperRobot:
    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig()

        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.piper_motors = make_motors_buses_from_configs(self.config.follower_arm)
        self.arm = self.piper_motors["main"]
        self.teleop_mode = "leader"

        if not self.inference_time:
            self.teleop = self._create_teleop_controller()
        else:
            self.teleop = None

        self.logs = {}
        self.is_connected = False

        # Optional policy compatibility mode: expose a 6D state/action interface
        # to the policy by dropping joint_4 (0-based index 3) while keeping the
        # physical robot command 7D and holding joint_4 at a fixed value.
        self.policy_compact_joint4 = os.getenv(
            "LEROBOT_PIPER_POLICY_DROP_JOINT4", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self.policy_disabled_joint_index = 3
        self._policy_joint_indices = [0, 1, 2, 4, 5, 6]
        self._fixed_joint4_value = None

        # policy limits (环境变量可配置, 解决 send_action 硬限幅导致的动作慢)
        #   LEROBOT_POLICY_JOINT_DELTA   每步关节最大变化 (rad), 默认 0.025
        #   LEROBOT_POLICY_GRIPPER_DELTA 每步夹爪最大变化, 默认 0.005
        # 注意: 调大可让推理/回摆动作更快, 但安全性下降 (防猛动撞机)
        self.policy_joint_delta_limit = float(os.getenv("LEROBOT_POLICY_JOINT_DELTA", "0.025"))
        self.policy_gripper_delta_limit = float(os.getenv("LEROBOT_POLICY_GRIPPER_DELTA", "0.005"))

        # teleop filters
        self.teleop_joint_deadbands = [0.01, 0.008, 0.008, 0.01, 0.003, 0.01]
        self.teleop_joint4_deadband = 0.003
        self.teleop_joint4_smoothing_alpha = 0.7
        self.teleop_target_alpha = [0.22, 0.14, 0.20, 0.20, 0.18, 0.20, 0.25]
        self.teleop_hysteresis = [0.006, 0.006, 0.006, 0.006, 0.008, 0.006, 0.002]
        self.teleop_settle_enter = [0.006, 0.0085, 0.010, 0.010, 0.007, 0.010, 0.004]
        self.teleop_settle_exit = [0.012, 0.0165, 0.018, 0.018, 0.014, 0.018, 0.007]
        self.teleop_follow_error_limit = [0.08, 0.05, 0.07, 0.07, 0.06, 0.05, 0.02]
        self.teleop_settle_frames = 3
        self.teleop_startup_hold_s = 2.0
        self.startup_zero_hold_s = float(os.getenv("LEROBOT_PIPER_STARTUP_ZERO_HOLD_S", "0.8"))

        # 遥控夹爪增益: leader 夹爪范围与 follower 不匹配时放大
        # 例: 1.2 → leader 满开时 follower 开到 120% (0.1→0.12)
        # 仅作用于遥控 (teleop), 不影响推理
        self.teleop_gripper_gain = float(os.getenv("LEROBOT_PIPER_TELEOP_GRIPPER_GAIN", "1.0"))
        # 上电回零时夹爪初始张开位置 (归一化上限 0.1)
        self.startup_gripper_open = float(os.getenv("LEROBOT_PIPER_STARTUP_GRIPPER_OPEN", "0.1"))

        # states
        self._last_joint4_target = None
        self._teleop_filtered_target = None
        self._teleop_command_state = None
        self._teleop_settle_counter = [0] * 7
        self._teleop_settled = [False] * 7
        self._last_sent_teleop_target = None
        self._last_sent_policy_target = None
        self._teleop_filter_armed = False
        self._teleop_filter_start_t = time.perf_counter()
        self._teleop_origin_state = None

        # debug / reference mode
        self.teleop_clip_reference_mode = os.getenv(
            "LEROBOT_TELEOP_CLIP_REFERENCE", "last_sent"
        ).strip().lower()
        self.policy_clip_reference_mode = os.getenv(
            "LEROBOT_POLICY_CLIP_REFERENCE", "current"
        ).strip().lower()

        self.debug_teleop_clip = os.getenv(
            "LEROBOT_DEBUG_TELEOP_CLIP", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self.debug_policy_clip = os.getenv(
            "LEROBOT_DEBUG_POLICY_CLIP", "0"
        ).strip().lower() in {"1", "true", "yes"}

        self._teleop_debug_counter = 0
        self._policy_debug_counter = 0

        # === 力传感器（可选）===
        self.force_sensor = None
        if _FORCE_AVAILABLE:
            try:
                self.force_sensor = ATIForceSensor(
                    cal_path=_force_cal,
                    device=os.getenv("LEROBOT_FORCE_DEVICE", "Dev1"),
                    channels=os.getenv("LEROBOT_FORCE_CHANNELS", "ai0:5"),
                )
            except Exception as e:
                print(f"[PiperRobot] 力传感器初始化失败: {e}")
                self.force_sensor = None

    def _create_teleop_controller(self):
        controller, detected_mode = create_piper_teleop_controller(prefer_leader=True)
        self.teleop_mode = detected_mode
        print(f"Teleop controller selected: {detected_mode}")
        return controller

    @property
    def has_force_sensor(self):
        return self.force_sensor is not None

    def _read_force(self) -> np.ndarray | None:
        """读取力传感器数据，如果未配置则返回 None"""
        if self.force_sensor is None:
            return None
        try:
            return self.force_sensor.read()
        except Exception as e:
            print(f"[PiperRobot] 力传感器读取失败: {e}")
            return None

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = get_motor_names(self.piper_motors)
        state_names = get_motor_names(self.piper_motors)
        if self.policy_compact_joint4:
            action_names = [action_names[idx] for idx in self._policy_joint_indices]
            state_names = [state_names[idx] for idx in self._policy_joint_indices]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def force_features(self) -> dict:
        """力传感器 feature 定义"""
        if self.force_sensor is None:
            return {}
        return {
            "observation.force": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            },
        }

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )

        self.arm.connect(enable=True)
        print("piper connected")

        if self.teleop is not None and hasattr(self.teleop, "connect"):
            self.teleop.connect()

        for name in self.cameras:
            self.cameras[name].connect()
            print(f"camera {name} connected")

        if self.force_sensor is not None:
            self.force_sensor.connect()

        print("All connected")
        self.is_connected = True
        self.run_calibration()

    def disconnect(self) -> None:
        """Move to home position, disable piper, and disconnect cameras."""
        if not self.inference_time and self.teleop is not None:
            self.teleop.stop()

        self.arm.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.arm.connect(enable=False)

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        if self.force_sensor is not None:
            self.force_sensor.close()

        self.is_connected = False

    def run_calibration(self):
        """Move piper to the home position."""
        if not self.is_connected:
            raise ConnectionError()

        print("Moving piper to startup zero pose...")
        self.arm.apply_calibration()
        # 上电回零后夹爪张开到 startup_gripper_open (默认 0.1)
        startup_home = list(self.arm.init_joint_position)
        startup_home[-1] = self.startup_gripper_open
        if hasattr(self.arm, "hold_position"):
            self.arm.hold_position(
                startup_home,
                hold_s=self.startup_zero_hold_s,
                interval_s=0.05,
            )

        self._last_joint4_target = None
        self._teleop_filtered_target = None
        self._teleop_command_state = None
        self._teleop_settle_counter = [0] * 7
        self._teleop_settled = [False] * 7
        self._last_sent_teleop_target = None
        self._last_sent_policy_target = None
        self._fixed_joint4_value = None
        self._teleop_filter_armed = False
        self._teleop_filter_start_t = time.perf_counter()
        self._teleop_origin_state = None

        if not self.inference_time and self.teleop is not None:
            self.teleop.reset()

    def _teleop_uses_relative_offsets(self) -> bool:
        return self.teleop_mode == "uarm"

    def _apply_teleop_origin(self, target_joints: list[float], current_state: list[float]) -> list[float]:
        if not self._teleop_uses_relative_offsets():
            return list(target_joints)

        if self._teleop_origin_state is None:
            self._teleop_origin_state = list(current_state)

        absolute_targets = list(target_joints)
        for idx in range(min(6, len(absolute_targets), len(self._teleop_origin_state))):
            absolute_targets[idx] = float(absolute_targets[idx] + self._teleop_origin_state[idx])

        # 夹爪: 开合增益 (leader 输出直接 × gain, 方向与原版一致)
        if len(absolute_targets) > 6:
            follower_target = float(target_joints[-1]) * self.teleop_gripper_gain
            absolute_targets[-1] = max(0.0, min(0.1, follower_target))

        return absolute_targets

    def _filter_uarm_teleop_targets(self, target_joints: list[float], current_state: list[float]) -> list[float]:
        raw_input_joints = list(target_joints)

        if self._teleop_filtered_target is None:
            self._teleop_filtered_target = list(current_state)
        if self._teleop_command_state is None:
            self._teleop_command_state = list(current_state)

        if not self._teleop_filter_armed:
            self._teleop_filter_armed = True
            self._teleop_filtered_target = list(current_state)
            self._teleop_command_state = list(current_state)
            self.logs["teleop_clip_reference_mode"] = "startup_hold"
            self.logs["teleop_clip_reference_state"] = list(current_state)
            return list(current_state)

        if time.perf_counter() - self._teleop_filter_start_t < self.teleop_startup_hold_s:
            self._teleop_filtered_target = list(current_state)
            self._teleop_command_state = list(current_state)
            self.logs["teleop_clip_reference_mode"] = "startup_hold"
            self.logs["teleop_clip_reference_state"] = list(current_state)
            return list(current_state)

        filtered_target_joints = []
        for idx, (raw, prev_filtered) in enumerate(
            zip(raw_input_joints, self._teleop_filtered_target, strict=True)
        ):
            alpha = self.teleop_target_alpha[idx] if idx < len(self.teleop_target_alpha) else 0.18
            filtered = (1.0 - alpha) * prev_filtered + alpha * raw
            filtered_target_joints.append(float(filtered))

        self._teleop_filtered_target = list(filtered_target_joints)
        self._teleop_command_state = list(filtered_target_joints)
        self._teleop_settle_counter = [0] * len(filtered_target_joints)
        self._teleop_settled = [False] * len(filtered_target_joints)

        self.logs["teleop_clip_reference_mode"] = "current_state_low_pass"
        self.logs["teleop_current_state"] = list(current_state)
        self.logs["teleop_clip_reference_state"] = list(current_state)
        self.logs["teleop_action_raw"] = list(raw_input_joints)
        self.logs["teleop_action_input"] = list(raw_input_joints)
        self.logs["teleop_action_filtered"] = list(filtered_target_joints)
        self.logs["teleop_action_clipped"] = list(filtered_target_joints)
        self.logs["teleop_action_settle_mask"] = list(self._teleop_settled)
        self.logs["teleop_settled_flags"] = list(self._teleop_settled)
        self.logs["teleop_settle_counter"] = list(self._teleop_settle_counter)
        self.logs["teleop_command_state"] = list(self._teleop_command_state)
        self.logs["teleop_command_prev"] = list(current_state)
        self.logs["teleop_command_new"] = list(filtered_target_joints)
        self.logs["teleop_err_current"] = [
            target - current for target, current in zip(filtered_target_joints, current_state, strict=True)
        ]
        self.logs["teleop_err_cmd"] = [
            target - current for target, current in zip(filtered_target_joints, current_state, strict=True)
        ]
        self.logs["teleop_step_limit"] = [float("nan")] * len(filtered_target_joints)
        self.logs["teleop_follow_limit"] = [float("nan")] * len(filtered_target_joints)
        self.logs["teleop_input_minus_reference"] = [
            t - r for t, r in zip(raw_input_joints, current_state, strict=True)
        ]
        self.logs["teleop_filtered_minus_reference"] = [
            t - r for t, r in zip(filtered_target_joints, current_state, strict=True)
        ]
        self.logs["teleop_clipped_minus_reference"] = [
            c - r for c, r in zip(filtered_target_joints, current_state, strict=True)
        ]
        self.logs["teleop_input_minus_current"] = [
            t - c for t, c in zip(raw_input_joints, current_state, strict=True)
        ]
        self.logs["teleop_filtered_minus_current"] = [
            t - c for t, c in zip(filtered_target_joints, current_state, strict=True)
        ]
        self.logs["teleop_clipped_minus_current"] = [
            c - s for c, s in zip(filtered_target_joints, current_state, strict=True)
        ]
        self._last_sent_teleop_target = list(filtered_target_joints)

        return filtered_target_joints

    def _compact_state_for_policy(self, values: list[float]) -> list[float]:
        if not self.policy_compact_joint4:
            return list(values)
        return [values[idx] for idx in self._policy_joint_indices]

    def _get_fixed_joint4_value(self, full_state: list[float]) -> float:
        if self._fixed_joint4_value is None:
            self._fixed_joint4_value = float(full_state[self.policy_disabled_joint_index])
        return self._fixed_joint4_value

    def _expand_policy_action(self, policy_action: list[float], full_state: list[float]) -> list[float]:
        if not self.policy_compact_joint4:
            return list(policy_action)

        if len(policy_action) != len(self._policy_joint_indices):
            raise ValueError(
                f"Expected {len(self._policy_joint_indices)}-D compact policy action, got {len(policy_action)}."
            )

        fixed_joint4 = self._get_fixed_joint4_value(full_state)
        expanded_action = []
        compact_idx = 0
        for full_idx in range(len(full_state)):
            if full_idx == self.policy_disabled_joint_index:
                expanded_action.append(fixed_joint4)
            else:
                expanded_action.append(float(policy_action[compact_idx]))
                compact_idx += 1
        return expanded_action

    def _store_action_trace(
        self,
        prefix: str,
        raw_target_joints: list[float],
        reference_state: list[float],
        current_state_before: list[float],
        clipped_action: list[float],
        state_after: list[float],
    ) -> None:
        self.logs[f"{prefix}_arm_read_before"] = list(current_state_before)
        self.logs[f"{prefix}_reference_state"] = list(reference_state)
        self.logs[f"{prefix}_raw_target_joints"] = list(raw_target_joints)
        self.logs[f"{prefix}_clipped_action"] = list(clipped_action)
        self.logs[f"{prefix}_arm_read_after"] = list(state_after)
        self.logs[f"{prefix}_action_raw"] = list(raw_target_joints)
        self.logs[f"{prefix}_action_clipped"] = list(clipped_action)

        self.logs[f"{prefix}_raw_minus_reference"] = [
            r - ref for r, ref in zip(raw_target_joints, reference_state)
        ]
        self.logs[f"{prefix}_clipped_minus_reference"] = [
            c - ref for c, ref in zip(clipped_action, reference_state)
        ]
        self.logs[f"{prefix}_current_minus_reference"] = [
            cur - ref for cur, ref in zip(current_state_before, reference_state)
        ]
        self.logs[f"{prefix}_after_minus_clipped"] = [
            s - c for s, c in zip(state_after, clipped_action)
        ]

        should_print = (
            (prefix == "teleop" and self.debug_teleop_clip)
            or (prefix == "policy" and self.debug_policy_clip)
        )
        if not should_print:
            return

        print(f"[{prefix}] read_before      =", np.round(current_state_before, 4).tolist())
        print(f"[{prefix}] reference_state =", np.round(reference_state, 4).tolist())
        print(f"[{prefix}] raw_target       =", np.round(raw_target_joints, 4).tolist())
        print(f"[{prefix}] clipped_action   =", np.round(clipped_action, 4).tolist())
        print(f"[{prefix}] read_after       =", np.round(state_after, 4).tolist())
        print(
            f"[{prefix}] raw-ref          =",
            np.round(self.logs[f"{prefix}_raw_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] clip-ref         =",
            np.round(self.logs[f"{prefix}_clipped_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] cur-ref          =",
            np.round(self.logs[f"{prefix}_current_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] after-clipped    =",
            np.round(self.logs[f"{prefix}_after_minus_clipped"], 4).tolist(),
        )
        print("=" * 60)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None and self.inference_time:
            self.teleop = self._create_teleop_controller()

        # 排空力传感器缓冲（防止连续采样溢出）
        if self.force_sensor is not None and hasattr(self.force_sensor, "drain"):
            self.force_sensor.drain()

        before_read_t = time.perf_counter()
        state_before_dict = self.arm.read()
        current_state_before = list(state_before_dict.values())
        action_dict_raw = self.teleop.get_action()
        raw_target_joints = self._apply_teleop_origin(list(action_dict_raw.values()), current_state_before)
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        control_mode = self.teleop.get_control_mode()
        self.logs["teleop_control_mode"] = control_mode

        executed_action = None
        state_after = list(current_state_before)
        need_post_write_read = record_data or self.debug_teleop_clip

        if control_mode == "ee":
            pose_target = self.teleop.get_pose_target()
            self.logs["teleop_pose_target"] = pose_target
            self.logs["teleop_gripper_target"] = getattr(self.teleop, "gripper", None)

            if pose_target is not None:
                self.arm.write_pose(pose_target, self.teleop.gripper)
                if need_post_write_read:
                    state_after = list(self.arm.read().values())
                self.logs["teleop_arm_read_before"] = list(current_state_before)
                self.logs["teleop_arm_read_after"] = list(state_after)
            else:
                executed_action = self._clip_teleop_targets(raw_target_joints, current_state_before)
                self.arm.write(executed_action)
                if need_post_write_read:
                    state_after = list(self.arm.read().values())
                reference_state = list(
                    self.logs.get("teleop_clip_reference_state", current_state_before)
                )
                self._store_action_trace(
                    "teleop",
                    raw_target_joints,
                    reference_state,
                    current_state_before,
                    executed_action,
                    state_after,
                )
        else:
            executed_action = self._clip_teleop_targets(raw_target_joints, current_state_before)
            self.arm.write(executed_action)
            if need_post_write_read:
                state_after = list(self.arm.read().values())
            reference_state = list(
                self.logs.get("teleop_clip_reference_state", current_state_before)
            )
            self._store_action_trace(
                "teleop",
                raw_target_joints,
                reference_state,
                current_state_before,
                executed_action,
                state_after,
            )

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return

        state_tensor = torch.as_tensor(
            self._compact_state_for_policy(current_state_before), dtype=torch.float32
        )
        if executed_action is None:
            action_tensor = torch.as_tensor(
                self._compact_state_for_policy(raw_target_joints), dtype=torch.float32
            )
        else:
            action_tensor = torch.as_tensor(
                self._compact_state_for_policy(executed_action), dtype=torch.float32
            )

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state_tensor
        action_dict["action"] = action_tensor
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        # === 力传感器（可选）===
        ft = self._read_force()
        if ft is not None:
            obs_dict["observation.force"] = torch.as_tensor(ft, dtype=torch.float32)

        return obs_dict, action_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Write the predicted actions from policy to the motors."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        compact_action = action.tolist()
        current_state_before = list(self.arm.read().values())
        raw_target_joints = self._expand_policy_action(compact_action, current_state_before)

        if (
            self.policy_clip_reference_mode == "last_sent"
            and self._last_sent_policy_target is not None
        ):
            reference_state = list(self._last_sent_policy_target)
        else:
            reference_state = list(current_state_before)

        clipped_action = []
        for idx, (target, reference) in enumerate(
            zip(raw_target_joints, reference_state, strict=True)
        ):
            max_delta = (
                self.policy_gripper_delta_limit
                if idx == len(raw_target_joints) - 1
                else self.policy_joint_delta_limit
            )
            clipped_action.append(float(np.clip(target, reference - max_delta, reference + max_delta)))

        self.arm.write(clipped_action)
        state_after = list(self.arm.read().values())

        self.logs["policy_clip_reference_mode"] = self.policy_clip_reference_mode
        self._store_action_trace(
            "policy",
            raw_target_joints,
            reference_state,
            current_state_before,
            clipped_action,
            state_after,
        )

        self._last_sent_policy_target = list(clipped_action)
        return torch.as_tensor(self._compact_state_for_policy(clipped_action), dtype=action.dtype)

    def _clip_teleop_targets(self, target_joints: list[float], current_state: list[float]) -> list[float]:
        if self._teleop_uses_relative_offsets():
            return self._filter_uarm_teleop_targets(target_joints, current_state)

        raw_input_joints = list(target_joints)
        clipped_action = []
        gripper_index = len(raw_input_joints) - 1
        joint4_index = 4
        if self.teleop_clip_reference_mode == "last_sent" and self._last_sent_teleop_target is not None:
            reference_state = self._last_sent_teleop_target
        else:
            reference_state = current_state

        joint4_target = raw_input_joints[joint4_index]
        if self._last_joint4_target is None:
            self._last_joint4_target = reference_state[joint4_index]

        if abs(joint4_target - reference_state[joint4_index]) < self.teleop_joint4_deadband:
            joint4_target = reference_state[joint4_index]
        else:
            joint4_target = (
                (1 - self.teleop_joint4_smoothing_alpha) * self._last_joint4_target
                + self.teleop_joint4_smoothing_alpha * joint4_target
            )

        raw_input_joints[joint4_index] = joint4_target
        self._last_joint4_target = joint4_target

        for idx, deadband in enumerate(self.teleop_joint_deadbands):
            if idx == joint4_index:
                continue
            if abs(raw_input_joints[idx] - reference_state[idx]) < deadband:
                raw_input_joints[idx] = reference_state[idx]

        for idx, (target, current) in enumerate(zip(raw_input_joints, reference_state, strict=True)):
            err = abs(target - current)

            if idx == gripper_index:
                if err > 0.02:
                    max_delta = 0.006
                else:
                    max_delta = 0.003
            elif idx in {1, 2}:
                if err > 0.20:
                    max_delta = 0.08
                elif err > 0.08:
                    max_delta = 0.05
                else:
                    max_delta = 0.025
            elif idx == joint4_index:
                if err > 0.12:
                    max_delta = 0.05
                elif err > 0.05:
                    max_delta = 0.03
                else:
                    max_delta = 0.015
            else:
                if err > 0.15:
                    max_delta = 0.07
                elif err > 0.06:
                    max_delta = 0.045
                else:
                    max_delta = 0.02

            clipped_action.append(float(np.clip(target, current - max_delta, current + max_delta)))

        self.logs["teleop_action_raw"] = list(raw_input_joints)
        self.logs["teleop_action_clipped"] = list(clipped_action)
        self.logs["teleop_clip_reference_mode"] = self.teleop_clip_reference_mode
        self.logs["teleop_clip_reference_state"] = list(reference_state)
        self._last_sent_teleop_target = list(clipped_action)

        if self.debug_teleop_clip:
            self._teleop_debug_counter += 1
            max_delta = max(abs(raw - clipped) for raw, clipped in zip(raw_input_joints, clipped_action, strict=True))
            if max_delta > 1e-6 and self._teleop_debug_counter % 10 == 0:
                raw_str = " ".join(f"j{i}={value:+.3f}" for i, value in enumerate(raw_input_joints))
                clipped_str = " ".join(f"j{i}={value:+.3f}" for i, value in enumerate(clipped_action))
                ref_str = " ".join(f"j{i}={value:+.3f}" for i, value in enumerate(reference_state))
                print(
                    f"[teleop-clip] mode={self.teleop_clip_reference_mode} max_delta={max_delta:.4f}\n"
                    f"  ref: {ref_str}\n"
                    f"  raw: {raw_str}\n"
                    f"  out: {clipped_str}"
                )

        return clipped_action

    def capture_observation(self) -> dict:
        """Capture current images and joint positions."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        before_read_t = time.perf_counter()
        state = self.arm.read()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        full_state = list(state.values())
        state = torch.as_tensor(self._compact_state_for_policy(full_state), dtype=torch.float32)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        # === 力传感器（可选）===
        ft = self._read_force()
        if ft is not None:
            obs_dict["observation.force"] = torch.as_tensor(ft, dtype=torch.float32)
        return obs_dict

    def consume_control_events(self) -> dict[str, bool]:
        if self.teleop is None:
            return {"exit_early": False, "rerecord_episode": False}
        return self.teleop.consume_control_events()

    def teleop_safety_stop(self):
        """Move to home position after recording one episode."""
        self.run_calibration()

    def __del__(self):
        if self.is_connected:
            self.disconnect()
            if not self.inference_time and self.teleop is not None:
                self.teleop.stop()
