import json
import math
import os
import time
import pygame
import threading
import re
from pathlib import Path
from typing import Dict
import serial

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

try:
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from trac_ik import TracIK
    IK_IMPORT_ERROR = None
except ImportError as exc:
    np = None
    R = None
    TracIK = None
    IK_IMPORT_ERROR = exc


SONY_WIRELESS_CONTROLLER_NAME = "Sony Computer Entertainment Wireless Controller"
DEADZONE = 0.28
AXIS_SMOOTHING_ALPHA = 0.25
JOINT_STEP_SCALE = 0.006
DPAD_STEP_SCALE = 0.006
GRIPPER_STEP_SCALE = 0.001

SONY_BUTTON_MAP = {
    "a": 0,
    "b": 1,
    "x": 3,
    "y": 2,
    "lb": 4,
    "rb": 5,
    "lt": 6,
    "rt": 7,
    "back": 8,
    "start": 9,
    "home": 10,
    "l3": 11,
    "r3": 12,
}

SONY_AXIS_MAP = {
    "left_x": 0,
    "left_y": 1,
    "left_trigger": 2,
    "right_x": 3,
    "right_y": 4,
    "right_trigger": 5,
}

SONY_HAT_MAP = {
    "dpad": 0,
}

DEFAULT_BUTTON_MAP = SONY_BUTTON_MAP
DEFAULT_AXIS_MAP = SONY_AXIS_MAP
DEFAULT_HAT_MAP = SONY_HAT_MAP

EE_SPEED_SCALE = 0.6
DEFAULT_SO101_LEADER_PORT = "/dev/ttyACM0"
DEFAULT_UARM_PORT = "/dev/ttyUSB0"
DEFAULT_SO101_CALIBRATION = Path(
    "/home/night/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/R07252801.json"
)
JOINT0_OFFSET_DEG = 100.0
JOINT1_OFFSET_DEG = 90.0
JOINT2_ZERO_DEG = -10.0
JOINT4_ZERO_DEG = 223.0
JOINT5_ZERO_DEG = 32.0
JOINT1_SMOOTHING_ALPHA = 0.35
PIPER_JOINT_LIMITS_RAD = [
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.967, 2.967),
]
PIPER_JOINT_LIMITS_DEG = [(math.degrees(lower), math.degrees(upper)) for lower, upper in PIPER_JOINT_LIMITS_RAD]
DEBUG_SO101_MAPPING = os.getenv("LEROBOT_DEBUG_SO101_MAPPING", "0") == "1"
DEBUG_PRINT_INTERVAL_S = 0.2
SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
DEFAULT_SO101_HOLD_JOINTS = ["shoulder_lift", "elbow_flex"]
DEFAULT_SO101_HOLD_INTERVAL_S = 0.12
DEFAULT_SO101_HOLD_REST_DEADBAND = 0.6
DEFAULT_SO101_HOLD_SETTLE_TIME_S = 2
DEFAULT_SO101_HOLD_TORQUE_LIMIT = 10
DEFAULT_SO101_HOLD_RETARGET_THRESHOLD = 8.0
DEFAULT_UARM_PIPER_JOINT_SCALE = (1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
DEFAULT_UARM_GRIPPER_OPEN_DEG = -7.0
DEFAULT_UARM_GRIPPER_CLOSED_DEG = 48.0
DEFAULT_UARM_GRIPPER_MIN = 0.0
DEFAULT_UARM_GRIPPER_MAX = 0.1  # piper 夹爪行程上限 (归一化 [0, 0.1])


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_motor_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return list(default)

    names = []
    for item in value.split(","):
        motor_name = item.strip()
        if motor_name and motor_name in SO101_MOTOR_NAMES and motor_name not in names:
            names.append(motor_name)
    return names


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float_tuple(name: str, default: tuple[float, ...], expected_len: int) -> tuple[float, ...]:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError:
        return default
    if len(parsed) != expected_len:
        return default
    return parsed


def _env_torque_scale_for_motor(motor_name: str) -> float:
    alias_map = {
        "elbow_flex": "L3",
        "wrist_flex": "L4",
    }
    alias = alias_map.get(motor_name)
    env_names = [f"LEROBOT_SO101_HOLD_TORQUE_SCALE_{motor_name.upper()}"]
    if alias is not None:
        env_names.insert(0, f"LEROBOT_SO101_HOLD_TORQUE_SCALE_{alias}")

    for env_name in env_names:
        value = os.getenv(env_name)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except ValueError:
            continue

    return 1.0


def _detect_default_leader_port() -> str:
    explicit = os.getenv("LEROBOT_SO101_PORT")
    if explicit:
        return explicit

    candidates = sorted(Path("/dev").glob("ttyACM*")) + sorted(Path("/dev").glob("ttyUSB*"))
    if candidates:
        return str(candidates[0])

    return DEFAULT_SO101_LEADER_PORT


def detect_so101_leader_port(port: str | None = None) -> str | None:
    candidate = port or os.getenv("LEROBOT_SO101_PORT")
    if candidate:
        return candidate if Path(candidate).exists() else None

    for detected in sorted(Path("/dev").glob("ttyACM*")) + sorted(Path("/dev").glob("ttyUSB*")):
        if detected.exists():
            return str(detected)

    return None


def detect_uarm_port(port: str | None = None) -> str | None:
    candidate = port or os.getenv("LEROBOT_UARM_PORT")
    if candidate:
        # Windows 串口路径 (COMx) 不是文件系统路径，直接返回
        if os.name == "nt" or candidate.startswith("COM"):
            return candidate
        return candidate if Path(candidate).exists() else None

    return DEFAULT_UARM_PORT if Path(DEFAULT_UARM_PORT).exists() else None


def _is_legacy_so101_calibration(calibration: dict) -> bool:
    required_keys = {"homing_offset", "drive_mode", "start_pos", "end_pos", "calib_mode", "motor_names"}
    return required_keys.issubset(calibration.keys())


def _is_full_so101_calibration(calibration: dict) -> bool:
    return all(name in calibration and isinstance(calibration[name], dict) for name in SO101_MOTOR_NAMES)


def _convert_full_so101_calibration(calibration: dict) -> dict:
    return {
        "homing_offset": [calibration[name]["homing_offset"] for name in SO101_MOTOR_NAMES],
        "drive_mode": [calibration[name]["drive_mode"] for name in SO101_MOTOR_NAMES],
        "start_pos": [calibration[name]["range_min"] for name in SO101_MOTOR_NAMES],
        "end_pos": [calibration[name]["range_max"] for name in SO101_MOTOR_NAMES],
        "calib_mode": ["DEGREE", "DEGREE", "DEGREE", "DEGREE", "DEGREE", "LINEAR"],
        "motor_names": SO101_MOTOR_NAMES,
    }


def _normalize_so101_calibration(calibration: dict) -> dict:
    if _is_legacy_so101_calibration(calibration):
        return calibration

    if _is_full_so101_calibration(calibration):
        return _convert_full_so101_calibration(calibration)

    raise ValueError("Unsupported SO101 calibration format.")


def _candidate_so101_calibration_paths() -> list[Path]:
    explicit = os.getenv("LEROBOT_SO101_CALIBRATION")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend(
        [
            DEFAULT_SO101_CALIBRATION,
            Path(".cache/calibration/so101/main_leader.json"),
            Path(".cache/calibration/so100/main_leader.json"),
            Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "so101_leader" / "main.json",
            Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "so100_leader" / "main.json",
        ]
    )
    return candidates


def _load_so101_calibration(calibration: str | os.PathLike | dict | None) -> tuple[dict | None, Path | None]:
    if isinstance(calibration, dict):
        return _normalize_so101_calibration(calibration), None

    if calibration is not None:
        path = Path(calibration).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Leader calibration file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return _normalize_so101_calibration(json.load(f)), path

    for candidate in _candidate_so101_calibration_paths():
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return _normalize_so101_calibration(json.load(f)), candidate

    return None, None


def detect_so101_leader_available(
    port: str | None = None,
    calibration: str | os.PathLike | dict | None = None,
) -> bool:
    detected_port = detect_so101_leader_port(port)
    if detected_port is None:
        return False

    loaded_calibration, _ = _load_so101_calibration(calibration)
    return loaded_calibration is not None


def detect_uarm_available(port: str | None = None) -> bool:
    return detect_uarm_port(port) is not None


def detect_gamepad_available() -> bool:
    try:
        pygame.init()
        pygame.joystick.init()
        return pygame.joystick.get_count() > 0
    except Exception:
        return False
    finally:
        try:
            pygame.joystick.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass


def _wrap_deg(value: float) -> float:
    """Wrap angles into [-180, 180) to keep leader cross-zero jumps continuous."""
    return (value + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class UArmServoReader:
    """Read 7-channel servo offsets from a UArm controller over a serial port."""

    def __init__(
        self,
        port: str = DEFAULT_UARM_PORT,
        baudrate: int = 115200,
        timeout: float = 0.1,
        command_delay_s: float = 0.008,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.command_delay_s = command_delay_s
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.zero_angles = [0.0] * 7
        self._latest_offsets = [0.0] * 7
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._init_servos()

    def close(self) -> None:
        self.stop_stream()
        if getattr(self, "ser", None) is not None and self.ser.is_open:
            self.ser.close()

    def send_command(self, cmd: str) -> str:
        self.ser.write(cmd.encode("ascii"))
        time.sleep(self.command_delay_s)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def pwm_to_angle(
        self,
        response_str: str,
        servo_num: int,
        pwm_min: int = 500,
        pwm_max: int = 2500,
        angle_range: float = 270.0,
    ) -> float | None:
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        return (pwm_val - pwm_min) / (pwm_max - pwm_min) * angle_range

    def _init_servos(self) -> None:
        self.send_command("#000PVER!")
        for idx in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f"#{idx:03d}PULK!")
            response = self.send_command(f"#{idx:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip(), idx)
            self.zero_angles[idx] = angle if angle is not None else 0.0

    def start_stream(self, *, num_interp: int = 5, step_size: float = 1.0, interp_sleep_s: float = 0.02) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            kwargs={
                "num_interp": num_interp,
                "step_size": step_size,
                "interp_sleep_s": interp_sleep_s,
            },
            daemon=True,
        )
        self._stream_thread.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

    def get_latest_offsets(self) -> list[float]:
        with self._lock:
            return list(self._latest_offsets)

    def _stream_loop(self, *, num_interp: int, step_size: float, interp_sleep_s: float) -> None:
        angle_offset = [0.0] * 7
        target_angle_offset = [0.0] * 7

        while not self._stop_event.is_set():
            for idx in range(7):
                response = self.send_command(f"#{idx:03d}PRAD!")
                angle = self.pwm_to_angle(response.strip(), idx)
                if angle is not None:
                    new_angle = angle - self.zero_angles[idx]
                    if abs(new_angle - target_angle_offset[idx]) > step_size:
                        target_angle_offset[idx] = new_angle

            for _ in range(num_interp):
                if self._stop_event.is_set():
                    break
                for idx in range(7):
                    delta = target_angle_offset[idx] - angle_offset[idx]
                    angle_offset[idx] += delta * 0.2
                with self._lock:
                    self._latest_offsets = list(angle_offset)
                time.sleep(interp_sleep_s)


def _unwrap_near_reference(value: float, reference: float) -> float:
    """Return the angle equivalent to value that is closest to reference."""
    candidates = (value - 360.0, value, value + 360.0)
    return min(candidates, key=lambda candidate: abs(candidate - reference))


class _SO101LeaderController:
    """Leader-arm reader that mimics the gamepad controller interface used by PiperRobot."""

    def __init__(self, port: str | None = None, calibration: str | os.PathLike | dict | None = None, use_degrees: bool = False):
        self.port = port or _detect_default_leader_port()
        self.use_degrees = use_degrees
        self.calibration, self.calibration_path = _load_so101_calibration(calibration)
        self.bus = None
        self.control_mode = "joint"
        self.pose_target = None
        self.gripper = 0.0
        self._last_debug_print_t = 0.0
        self._last_debug_signature = None
        self._last_joint2_deg = None
        self._last_joint1_deg = None
        self._hold_torque_enabled = _env_flag("LEROBOT_SO101_HOLD_TORQUE", default=False)
        self._hold_motor_names = _env_motor_list("LEROBOT_SO101_HOLD_JOINTS", DEFAULT_SO101_HOLD_JOINTS)
        self._hold_interval_s = max(0.02, _env_float("LEROBOT_SO101_HOLD_INTERVAL_S", DEFAULT_SO101_HOLD_INTERVAL_S))
        self._hold_rest_deadband = max(
            0.0, _env_float("LEROBOT_SO101_HOLD_REST_DEADBAND", DEFAULT_SO101_HOLD_REST_DEADBAND)
        )
        self._hold_settle_time_s = max(
            0.0, _env_float("LEROBOT_SO101_HOLD_SETTLE_TIME_S", DEFAULT_SO101_HOLD_SETTLE_TIME_S)
        )
        self._hold_torque_limit = int(
            max(1, _env_float("LEROBOT_SO101_HOLD_TORQUE_LIMIT", DEFAULT_SO101_HOLD_TORQUE_LIMIT))
        )
        self._hold_torque_limits = {
            motor_name: max(1, int(round(self._hold_torque_limit * _env_torque_scale_for_motor(motor_name))))
            for motor_name in self._hold_motor_names
        }
        self._hold_retarget_threshold = max(
            0.0, _env_float("LEROBOT_SO101_HOLD_RETARGET_THRESHOLD", DEFAULT_SO101_HOLD_RETARGET_THRESHOLD)
        )
        self._hold_targets: dict[str, float] = {}
        self._last_hold_write_t = 0.0
        self._last_hold_state: dict[str, float] = {}
        self._last_hold_motion_t = 0.0
        self._hold_brake_engaged = False
        self._hold_current_torque_scale: float | None = None
        self._hold_current_uses_joint_scales: bool | None = None
        self._pending_events = {
            "exit_early": False,
            "rerecord_episode": False,
        }
        self._keyboard_listener = None

        if PYNPUT_AVAILABLE:
            self._keyboard_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
            self._keyboard_listener.start()

    def _on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char == "q":
                self._pending_events["exit_early"] = True
                return False
        except Exception:
            return None
        return None

    def connect(self):
        if self.bus is not None and self.bus.is_connected:
            return

        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        config = FeetechMotorsBusConfig(
            port=self.port,
            motors={
                "shoulder_pan": (1, "sts3215"),
                "shoulder_lift": (2, "sts3215"),
                "elbow_flex": (3, "sts3215"),
                "wrist_flex": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            },
        )
        self.bus = FeetechMotorsBus(config)
        self.bus.connect()
        print(f"leader connected on {self.port}")

        if self.calibration is None:
            raise FileNotFoundError(
                "SO101 leader calibration file was not found. "
                "Set LEROBOT_SO101_CALIBRATION or place a calibration file in .cache/calibration/so100/main_leader.json."
            )

        self.bus.set_calibration(self.calibration)
        self._configure_hold_torque()

    def _configure_hold_torque(self) -> None:
        if not self._hold_torque_enabled or self.bus is None or not self._hold_motor_names:
            return

        self.bus.write(
            "Torque_Limit",
            [self._hold_torque_limits[name] for name in self._hold_motor_names],
            self._hold_motor_names,
        )
        current_positions = self.bus.read("Present_Position", self._hold_motor_names)
        if hasattr(current_positions, "tolist"):
            current_positions = current_positions.tolist()
        self._hold_targets = dict(zip(self._hold_motor_names, current_positions, strict=True))
        self._last_hold_state = dict(self._hold_targets)
        self._last_hold_motion_t = time.perf_counter()
        self._set_hold_state(False, 0.0)
        self._last_hold_write_t = 0.0
        print(
            "[so101-hold] enabled "
            f"joints={','.join(self._hold_motor_names)} "
            f"interval={self._hold_interval_s:.3f}s "
            f"torque_limits={','.join(f'{name}:{self._hold_torque_limits[name]}' for name in self._hold_motor_names)} "
            f"rest_deadband={self._hold_rest_deadband:.2f} "
            f"settle={self._hold_settle_time_s:.2f}s"
        )

    def _write_hold_torque_limits(self, scale: float = 1.0) -> None:
        if self.bus is None or not self._hold_motor_names:
            return

        scale = _clamp(scale, 0.0, 1.0)
        self.bus.write(
            "Torque_Limit",
            [max(0, int(round(self._hold_torque_limits[name] * scale))) for name in self._hold_motor_names],
            self._hold_motor_names,
        )

    def _set_hold_state(
        self,
        engaged: bool,
        scale: float,
        goal_positions: list[float] | None = None,
        use_joint_scales: bool = True,
    ) -> None:
        if self.bus is None or not self._hold_motor_names:
            return

        scale = _clamp(scale, 0.0, 1.0)
        state_unchanged = self._hold_brake_engaged == engaged
        scale_unchanged = self._hold_current_torque_scale is not None and abs(self._hold_current_torque_scale - scale) < 1e-6
        mode_unchanged = self._hold_current_uses_joint_scales == use_joint_scales
        goals_unchanged = goal_positions is None
        if state_unchanged and scale_unchanged and mode_unchanged and goals_unchanged:
            return

        if engaged:
            if use_joint_scales:
                self._write_hold_torque_limits(scale)
            if goal_positions is not None:
                self.bus.write("Goal_Position", goal_positions, self._hold_motor_names)
            if not self._hold_brake_engaged:
                self.bus.write("Torque_Enable", 1, self._hold_motor_names)
        else:
            if self._hold_brake_engaged:
                self.bus.write("Torque_Enable", 0, self._hold_motor_names)

        self._hold_brake_engaged = engaged
        self._hold_current_torque_scale = scale if engaged else None
        self._hold_current_uses_joint_scales = use_joint_scales if engaged else None

    def _update_hold_targets(self, state: dict[str, float]) -> None:
        if not self._hold_torque_enabled or self.bus is None or not self._hold_targets:
            return

        now = time.perf_counter()
        moving = False
        goal_positions = []
        for motor_name in self._hold_motor_names:
            current_position = state[motor_name]
            last_position = self._last_hold_state.get(motor_name, current_position)
            motion_delta = abs(current_position - last_position)
            if motion_delta > self._hold_rest_deadband:
                moving = True
            goal_positions.append(current_position)

        self._last_hold_state = {name: state[name] for name in self._hold_motor_names}

        if moving:
            self._last_hold_motion_t = now
            self._hold_targets = dict(zip(self._hold_motor_names, goal_positions, strict=True))
            self._set_hold_state(False, 0.0)
            return

        if now - self._last_hold_motion_t < self._hold_settle_time_s:
            self._set_hold_state(False, 0.0)
            return

        if now - self._last_hold_write_t < self._hold_interval_s and self._hold_brake_engaged:
            return

        target_drift = max(
            abs(goal - self._hold_targets.get(name, goal))
            for name, goal in zip(self._hold_motor_names, goal_positions, strict=True)
        )
        if not self._hold_brake_engaged or target_drift > self._hold_retarget_threshold:
            self._hold_targets = dict(zip(self._hold_motor_names, goal_positions, strict=True))
            self._set_hold_state(True, 1.0, goal_positions)
            self._last_hold_write_t = now

    def _read_leader_state(self) -> dict[str, float]:
        if self.bus is None or not self.bus.is_connected:
            self.connect()

        values = self.bus.read("Present_Position")
        if hasattr(values, "tolist"):
            values = values.tolist()

        state = dict(zip(self.bus.motor_names, values, strict=True))
        self._update_hold_targets(state)
        return state

    def _map_so101_to_piper(self, state: dict[str, float]) -> Dict:
        # Feetech leader readings are calibrated to degrees for rotational joints and [0, 100] for the gripper.
        joint1_deg_raw = state["shoulder_lift"] - JOINT1_OFFSET_DEG
        if self._last_joint1_deg is None:
            joint1_deg = joint1_deg_raw
        else:
            joint1_deg = (
                (1.0 - JOINT1_SMOOTHING_ALPHA) * self._last_joint1_deg
                + JOINT1_SMOOTHING_ALPHA * joint1_deg_raw
            )
        self._last_joint1_deg = joint1_deg

        elbow_deg_raw = state["elbow_flex"] - JOINT2_ZERO_DEG
        if self._last_joint2_deg is None:
            elbow_deg = _wrap_deg(elbow_deg_raw)
        else:
            elbow_deg = _unwrap_near_reference(elbow_deg_raw, self._last_joint2_deg)
        elbow_deg = _clamp(elbow_deg, *PIPER_JOINT_LIMITS_DEG[2])
        self._last_joint2_deg = elbow_deg
        joint_deg = [
            -(state["shoulder_pan"] - JOINT0_OFFSET_DEG),
            joint1_deg,
            elbow_deg,
            0.0,
            _wrap_deg(state["wrist_flex"] - JOINT4_ZERO_DEG),
            -_wrap_deg(state["wrist_roll"] - JOINT5_ZERO_DEG),
        ]
        joint_deg = [
            _clamp(value, lower_deg, upper_deg)
            for value, (lower_deg, upper_deg) in zip(joint_deg, PIPER_JOINT_LIMITS_DEG, strict=True)
        ]
        joint_rad = [math.radians(value) for value in joint_deg]
        gripper_m = max(0.0, min(0.1, state["gripper"] / 100.0 * 0.1))
        self.gripper = gripper_m

        return {
            "joint0": joint_rad[0],
            "joint1": joint_rad[1],
            "joint2": joint_rad[2],
            "joint3": joint_rad[3],
            "joint4": joint_rad[4],
            "joint5": joint_rad[5],
            "gripper": gripper_m,
        }

    def _debug_print_mapping(self, state: dict[str, float], action: dict[str, float]) -> None:
        if not DEBUG_SO101_MAPPING:
            return

        signature = tuple(round(state[name], 1) for name in SO101_MOTOR_NAMES)
        now = time.perf_counter()
        if signature == self._last_debug_signature and now - self._last_debug_print_t < DEBUG_PRINT_INTERVAL_S:
            return

        joint_deg = {f"joint{i}": round(math.degrees(action[f"joint{i}"]), 1) for i in range(6)}
        state_fmt = " ".join(f"{name}={state[name]:+.1f}" for name in SO101_MOTOR_NAMES)
        action_fmt = " ".join(f"{name}={value:+.1f}" for name, value in joint_deg.items())
        print(f"[so101-state] {state_fmt}")
        print(f"[piper-map] {action_fmt} gripper={action['gripper']:.4f}")

        self._last_debug_signature = signature
        self._last_debug_print_t = now

    def get_action(self) -> Dict:
        state = self._read_leader_state()
        action = self._map_so101_to_piper(state)
        self._debug_print_mapping(state, action)
        return action

    def get_control_mode(self) -> str:
        return self.control_mode

    def get_pose_target(self):
        return None

    def consume_control_events(self) -> dict[str, bool]:
        return dict(self._pending_events)

    def go_home(self):
        pass

    def reset(self):
        pass

    def stop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self.bus is not None and self.bus.is_connected:
            if self._hold_torque_enabled and self._hold_motor_names:
                try:
                    self._set_hold_state(False, 0.0)
                except Exception:
                    pass
            self.bus.disconnect()


class _UArmLeaderController:
    """UArm reader that exposes the same teleop interface PiperRobot expects."""

    def __init__(
        self,
        port: str | None = None,
        baudrate: int | None = None,
        timeout: float | None = None,
        command_delay_s: float | None = None,
        interp_steps: int | None = None,
        interp_sleep_s: float | None = None,
        step_size_deg: float | None = None,
        joint_scale: tuple[float, ...] | None = None,
        gripper_open_deg: float | None = None,
        gripper_closed_deg: float | None = None,
        gripper_min: float | None = None,
        gripper_max: float | None = None,
    ):
        self.port = port or detect_uarm_port() or DEFAULT_UARM_PORT
        self.baudrate = baudrate if baudrate is not None else _env_int("LEROBOT_UARM_BAUDRATE", 115200)
        self.timeout = timeout if timeout is not None else _env_float("LEROBOT_UARM_TIMEOUT_S", 0.1)
        self.command_delay_s = (
            command_delay_s
            if command_delay_s is not None
            else _env_float("LEROBOT_UARM_COMMAND_DELAY_S", 0.008)
        )
        self.interp_steps = interp_steps if interp_steps is not None else _env_int("LEROBOT_UARM_INTERP_STEPS", 5)
        self.interp_sleep_s = (
            interp_sleep_s
            if interp_sleep_s is not None
            else _env_float("LEROBOT_UARM_INTERP_SLEEP_S", 0.02)
        )
        self.step_size_deg = (
            step_size_deg
            if step_size_deg is not None
            else _env_float("LEROBOT_UARM_STEP_SIZE_DEG", 1.0)
        )
        self.joint_scale = joint_scale or _env_float_tuple(
            "LEROBOT_UARM_JOINT_SCALE",
            DEFAULT_UARM_PIPER_JOINT_SCALE,
            expected_len=6,
        )
        self.gripper_open_deg = (
            gripper_open_deg
            if gripper_open_deg is not None
            else _env_float("LEROBOT_UARM_GRIPPER_OPEN_DEG", DEFAULT_UARM_GRIPPER_OPEN_DEG)
        )
        self.gripper_closed_deg = (
            gripper_closed_deg
            if gripper_closed_deg is not None
            else _env_float("LEROBOT_UARM_GRIPPER_CLOSED_DEG", DEFAULT_UARM_GRIPPER_CLOSED_DEG)
        )
        self.gripper_min = (
            gripper_min
            if gripper_min is not None
            else _env_float("LEROBOT_UARM_GRIPPER_MIN", DEFAULT_UARM_GRIPPER_MIN)
        )
        self.gripper_max = (
            gripper_max
            if gripper_max is not None
            else _env_float("LEROBOT_UARM_GRIPPER_MAX", DEFAULT_UARM_GRIPPER_MAX)
        )
        self.reader: UArmServoReader | None = None
        self.control_mode = "joint"
        self.pose_target = None
        self.gripper = self.gripper_max  # 初始上电夹爪张开到范围上限 (0.1)
        self._pending_events = {
            "exit_early": False,
            "rerecord_episode": False,
        }
        self._keyboard_listener = None

        if PYNPUT_AVAILABLE:
            self._keyboard_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
            self._keyboard_listener.start()

    def _on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char == "q":
                self._pending_events["exit_early"] = True
                return False
        except Exception:
            return None
        return None

    def _map_offsets_to_piper(self, offsets_deg: list[float]) -> Dict:
        if len(offsets_deg) < 7:
            raise ValueError(f"Expected 7 UArm servo offsets, got {len(offsets_deg)}.")

        joint_targets = [0.0] * 6

        for idx in range(6):
            joint_delta_rad = math.radians(float(offsets_deg[idx]) * float(self.joint_scale[idx]))
            joint_targets[idx] = joint_delta_rad

        gripper_angle = -float(offsets_deg[6])
        denom = self.gripper_closed_deg - self.gripper_open_deg
        if abs(denom) < 1e-6:
            raise ValueError("UArm gripper_open_deg and gripper_closed_deg must not be equal.")
        ratio = (self.gripper_closed_deg - gripper_angle) / denom
        self.gripper = _clamp(ratio, 0.0, 1.0) * (self.gripper_max - self.gripper_min) + self.gripper_min

        return {
            "joint0": joint_targets[0],
            "joint1": joint_targets[1],
            "joint2": joint_targets[2],
            "joint3": joint_targets[3],
            "joint4": joint_targets[4],
            "joint5": joint_targets[5],
            "gripper": self.gripper,
        }

    def connect(self):
        if self.reader is not None:
            return

        self.reader = UArmServoReader(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            command_delay_s=self.command_delay_s,
        )
        self.reader.start_stream(
            num_interp=self.interp_steps,
            step_size=self.step_size_deg,
            interp_sleep_s=self.interp_sleep_s,
        )
        print(f"UArm teleop connected on {self.port}")

    def get_action(self) -> Dict:
        if self.reader is None:
            self.connect()
        if self.reader is None:
            raise RuntimeError("UArm reader failed to initialize.")
        offsets = self.reader.get_latest_offsets()
        return self._map_offsets_to_piper(offsets)

    def get_control_mode(self) -> str:
        return self.control_mode

    def get_pose_target(self):
        return None

    def consume_control_events(self) -> dict[str, bool]:
        events = dict(self._pending_events)
        self._pending_events = {key: False for key in self._pending_events}
        return events

    def go_home(self):
        return None

    def reset(self):
        return None

    def stop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None


def SixAxisArmController_101(
    port: str | None = None,
    calibration: str | os.PathLike | dict | None = None,
    use_degrees: bool = False,
):
    return _SO101LeaderController(port=port, calibration=calibration, use_degrees=use_degrees)


def UArmController(port: str | None = None):
    return _UArmLeaderController(port=port)


def create_piper_teleop_controller(
    prefer_leader: bool = True,
    leader_port: str | None = None,
    leader_calibration: str | os.PathLike | dict | None = None,
    uarm_port: str | None = None,
):
    source = _env_str("LEROBOT_PIPER_TELEOP_SOURCE", "auto").lower()
    leader_available = detect_so101_leader_available(leader_port, leader_calibration)
    uarm_available = detect_uarm_available(uarm_port)
    gamepad_available = detect_gamepad_available()

    if source in {"uarm", "uarm_leader"}:
        controller = UArmController(port=uarm_port)
        return controller, "uarm"

    if source in {"so101", "leader", "so101_leader"}:
        controller = SixAxisArmController_101(port=leader_port, calibration=leader_calibration)
        return controller, "leader"

    if source == "gamepad":
        controller = SixAxisArmController()
        return controller, "gamepad"

    if prefer_leader and leader_available:
        controller = SixAxisArmController_101(port=leader_port, calibration=leader_calibration)
        return controller, "leader"

    if uarm_available:
        controller = UArmController(port=uarm_port)
        return controller, "uarm"

    if gamepad_available:
        controller = SixAxisArmController()
        return controller, "gamepad"

    if leader_available:
        controller = SixAxisArmController_101(port=leader_port, calibration=leader_calibration)
        return controller, "leader"

    if uarm_available:
        controller = UArmController(port=uarm_port)
        return controller, "uarm"

    controller = SixAxisArmController_101(port=leader_port, calibration=leader_calibration)
    return controller, "leader"


class PiperTracIKKinematics:
    def __init__(
        self,
        urdf_path: str,
        base_link_name: str = "base_link",
        target_link_name: str = "link6",
        timeout: float = 0.005,
        epsilon: float = 0.00001,
        solver_type: str = "Speed",
    ):
        self.ik_solver = TracIK(
            base_link_name=base_link_name,
            tip_link_name=target_link_name,
            urdf_path=urdf_path,
            timeout=timeout,
            epsilon=epsilon,
            solver_type=solver_type,
        )
        lower_limits, upper_limits = self.ik_solver.joint_limits
        self.joint_limits = list(zip(lower_limits.tolist(), upper_limits.tolist()))

    def solve_fk(self, joint_angles: list[float]) -> "np.ndarray":
        joint_array = np.asarray(joint_angles, dtype=float)
        position, rotation_matrix = self.ik_solver.fk(joint_array)
        rotation = R.from_matrix(rotation_matrix)
        return np.concatenate((position, rotation.as_euler("xyz", degrees=True)))

    def solve_ik(
        self,
        target_position: "np.ndarray",
        target_euler_deg: "np.ndarray",
        initial_guess: list[float],
    ) -> list[float] | None:
        target_rotation = R.from_euler("xyz", target_euler_deg, degrees=True)
        solution = self.ik_solver.ik(
            target_position,
            target_rotation.as_matrix(),
            seed_jnt_values=np.asarray(initial_guess, dtype=float),
        )
        if solution is None:
            return None
        return solution.tolist()


class SixAxisArmController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise Exception("未检测到手柄")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.device_name = self.joystick.get_name()
        self.button_map, self.axis_map, self.hat_map = self._select_maps(self.device_name)

        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper = 0.0
        self.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_speed = 0.0
        self.control_mode = "joint"
        # End-effector mode moves more stably with a reduced step size.
        self.pose_translation_step = 0.001 * EE_SPEED_SCALE
        self.pose_rotation_step_deg = 0.5 * EE_SPEED_SCALE
        self.pose_target = None
        self._last_debug_snapshot = None
        self._axis_filters = {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        }
        self._last_y_pressed = False
        self._last_start_pressed = False
        self._last_lb_pressed = False
        self._last_rb_pressed = False
        self._pending_events = {
            "exit_early": False,
            "rerecord_episode": False,
        }

        # Keep software limits aligned with the Piper URDF so EE IK clipping
        # matches the actual kinematic model used by the reference teleop code.
        self.joint_limits = [
            (-2.618, 2.618),
            (0.0, 3.14),
            (-2.967, 0.0),
            (-1.745, 1.745),
            (-1.22, 1.22),
            (-2.967, 2.967),
        ]
        self.kinematic = self._init_kinematics()
        if self.kinematic is not None:
            self._sync_pose_target_from_joints()

        self.running = True
        self.thread = threading.Thread(target=self.update_joints)
        self.thread.start()

    def _select_maps(self, device_name: str):
        if device_name == SONY_WIRELESS_CONTROLLER_NAME:
            return SONY_BUTTON_MAP, SONY_AXIS_MAP, SONY_HAT_MAP
        return DEFAULT_BUTTON_MAP, DEFAULT_AXIS_MAP, DEFAULT_HAT_MAP

    def _get_button(self, name: str) -> int:
        return self.joystick.get_button(self.button_map[name])

    def _get_axis(self, name: str, invert: bool = False) -> float:
        raw_value = self.joystick.get_axis(self.axis_map[name])
        raw_value = -raw_value if invert else raw_value
        if abs(raw_value) < DEADZONE:
            if name in self._axis_filters:
                self._axis_filters[name] = 0.0
            return 0.0

        if name in self._axis_filters:
            filtered = (1 - AXIS_SMOOTHING_ALPHA) * self._axis_filters[name] + AXIS_SMOOTHING_ALPHA * raw_value
            if abs(filtered) < 0.01:
                filtered = 0.0
            self._axis_filters[name] = filtered
            return filtered

        return raw_value

    def _get_hat(self, name: str) -> tuple[int, int]:
        return self.joystick.get_hat(self.hat_map[name])

    def _discover_urdf_path(self) -> str | None:
        candidates = [
            os.getenv("PIPER_URDF_PATH"),
            "/home/night/Gamepad_PiPER/piper/piper.urdf",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _init_kinematics(self) -> PiperTracIKKinematics | None:
        if np is None or R is None or TracIK is None:
            detail = f" ({IK_IMPORT_ERROR})" if IK_IMPORT_ERROR is not None else ""
            print(f"Piper EE mode unavailable: missing numpy/scipy/trac_ik dependencies{detail}.")
            return None

        urdf_path = self._discover_urdf_path()
        if urdf_path is None:
            print("Piper EE mode unavailable: URDF file not found. Set PIPER_URDF_PATH to enable it.")
            return None

        try:
            print(f"Initializing Piper EE mode with URDF: {urdf_path}")
            return PiperTracIKKinematics(urdf_path=urdf_path, base_link_name="base_link", target_link_name="link6")
        except Exception as exc:
            print(f"Piper EE mode unavailable: failed to initialize TracIK ({exc}).")
            return None

    def _sync_pose_target_from_joints(self) -> None:
        if self.kinematic is None:
            self.pose_target = None
            return
        try:
            self.pose_target = self.kinematic.solve_fk(self.joints)
        except Exception as exc:
            print(f"Failed to sync EE pose from joints: {exc}")
            self.pose_target = None

    def _toggle_control_mode(self) -> None:
        if self.control_mode == "joint":
            if self.kinematic is None:
                print("Cannot switch to EE mode: kinematics is unavailable.")
                return
            self._sync_pose_target_from_joints()
            if self.pose_target is None:
                print("Cannot switch to EE mode: failed to compute current end-effector pose.")
                return
            self.control_mode = "ee"
            print("Switched control mode to EE.")
            return

        self.control_mode = "joint"
        print("Switched control mode to joint.")

    def _debug_input_and_mode(
        self,
        start_pressed: bool,
        y_pressed: bool,
        lb_pressed: bool,
        rb_pressed: bool,
    ) -> None:
        snapshot = (
            start_pressed,
            y_pressed,
            lb_pressed,
            rb_pressed,
            self.control_mode,
            self.kinematic is not None,
        )
        if snapshot == self._last_debug_snapshot:
            return

        self._last_debug_snapshot = snapshot
        print(
            "[gamepad-debug] "
            f"start={int(start_pressed)} y={int(y_pressed)} "
            f"lb={int(lb_pressed)} rb={int(rb_pressed)} "
            f"mode={self.control_mode} "
            f"ee_ready={self.kinematic is not None}"
        )

    def _update_joint_control(
        self,
        left_x: float,
        left_y: float,
        right_x: float,
        right_y: float,
        up: bool,
        down: bool,
        left: bool,
        right: bool,
    ) -> None:
        target_speeds = [0.0] * 6
        target_speeds[0] = left_x * JOINT_STEP_SCALE
        target_speeds[1] = left_y * JOINT_STEP_SCALE
        target_speeds[2] = -right_y * JOINT_STEP_SCALE
        target_speeds[3] = right_x * JOINT_STEP_SCALE
        target_speeds[4] = -DPAD_STEP_SCALE if up else (DPAD_STEP_SCALE if down else 0.0)
        target_speeds[5] = DPAD_STEP_SCALE if right else (-DPAD_STEP_SCALE if left else 0.0)

        self.speeds = target_speeds

        for i in range(6):
            self.joints[i] += self.speeds[i]

        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            self.joints[i] = max(min_val, min(max_val, self.joints[i]))

    def _update_ee_control(
        self,
        left_x: float,
        left_y: float,
        right_x: float,
        right_y: float,
        up: bool,
        down: bool,
        left: bool,
        right: bool,
    ) -> None:
        if self.kinematic is None:
            return

        if self.pose_target is None:
            self._sync_pose_target_from_joints()
            if self.pose_target is None:
                return

        hat_x = 1 if right else (-1 if left else 0)
        hat_y = 1 if up else (-1 if down else 0)
        d_local = np.array([-left_y, left_x, -right_y], dtype=float) * self.pose_translation_step
        r_local = np.array([hat_x, -hat_y, right_x], dtype=float) * self.pose_rotation_step_deg

        if not np.any(d_local) and not np.any(r_local):
            self.speeds = [0.0] * 6
            return

        current_position = self.pose_target[:3]
        current_rotation = R.from_euler("xyz", self.pose_target[3:], degrees=True)
        target_rotation = current_rotation * R.from_euler("xyz", r_local, degrees=True)
        target_position = current_position + current_rotation.apply(d_local)
        ik_solution = self.kinematic.solve_ik(
            target_position=target_position,
            target_euler_deg=target_rotation.as_euler("xyz", degrees=True),
            initial_guess=self.joints,
        )
        if ik_solution is None:
            return

        old_joints = list(self.joints)
        self.joints = ik_solution
        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            self.joints[i] = max(min_val, min(max_val, self.joints[i]))
            self.speeds[i] = self.joints[i] - old_joints[i]

        self._sync_pose_target_from_joints()

    def consume_control_events(self) -> dict[str, bool]:
        events = dict(self._pending_events)
        self._pending_events = {key: False for key in self._pending_events}
        return events

    def update_joints(self):
        while self.running:
            try:
                pygame.event.pump()
            except Exception:
                self.stop()
                continue

            left_x = self._get_axis("left_x", invert=True)
            left_y = self._get_axis("left_y", invert=True)
            right_x = self._get_axis("right_x", invert=True)
            right_y = self._get_axis("right_y", invert=True)

            hat = self._get_hat("dpad")
            up = hat[1] == 1
            down = hat[1] == -1
            left = hat[0] == -1
            right = hat[0] == 1

            left_trigger = self._get_button("lt")
            right_trigger = self._get_button("rt")
            y_pressed = bool(self._get_button("y"))
            start_pressed = bool(self._get_button("start"))
            lb_pressed = bool(self._get_button("lb"))
            rb_pressed = bool(self._get_button("rb"))

            self._debug_input_and_mode(start_pressed, y_pressed, lb_pressed, rb_pressed)

            if rb_pressed and not self._last_rb_pressed:
                print("[gamepad-event] RB pressed -> exit current phase early")
                self._pending_events["exit_early"] = True

            if lb_pressed and not self._last_lb_pressed:
                print("[gamepad-event] LB pressed -> rerecord current episode")
                self._pending_events["rerecord_episode"] = True
                self._pending_events["exit_early"] = True

            if start_pressed and not self._last_start_pressed:
                self._toggle_control_mode()

            if y_pressed and not self._last_y_pressed:
                self.go_home()
                self._last_y_pressed = True
                self._last_start_pressed = start_pressed
                self._last_lb_pressed = lb_pressed
                self._last_rb_pressed = rb_pressed
                time.sleep(0.02)
                continue

            self._last_y_pressed = y_pressed
            self._last_start_pressed = start_pressed
            self._last_lb_pressed = lb_pressed
            self._last_rb_pressed = rb_pressed

            if self.control_mode == "joint":
                self._update_joint_control(left_x, left_y, right_x, right_y, up, down, left, right)
            else:
                self._update_ee_control(left_x, left_y, right_x, right_y, up, down, left, right)

            self.gripper_speed = (
                GRIPPER_STEP_SCALE if right_trigger else (-GRIPPER_STEP_SCALE if left_trigger else 0.0)
            )

            self.gripper += self.gripper_speed

            self.gripper = max(0.0, min(0.1, self.gripper))
            time.sleep(0.005)

    def get_action(self) -> Dict:
        return {
            "joint0": self.joints[0],
            "joint1": self.joints[1],
            "joint2": self.joints[2],
            "joint3": self.joints[3],
            "joint4": self.joints[4],
            "joint5": self.joints[5],
            "gripper": self.gripper,
        }

    def get_control_mode(self) -> str:
        return self.control_mode

    def get_pose_target(self):
        return None if self.pose_target is None else self.pose_target.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        pygame.quit()
        print("Gamepad exits")

    def go_home(self):
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper = 0.0
        self.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_speed = 0.0
        self._sync_pose_target_from_joints()

    def reset(self):
        self.go_home()


if __name__ == "__main__":
    arm_controller = SixAxisArmController()
    try:
        while True:
            print(arm_controller.get_action())
            print(arm_controller.consume_control_events())
            time.sleep(0.1)
    except KeyboardInterrupt:
        arm_controller.stop()
