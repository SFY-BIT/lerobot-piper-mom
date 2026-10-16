import time
import os
from typing import Dict
from piper_sdk import *
from piper_sdk.hardware_port.can_encapsulation_v0_4_0 import C_STD_CAN as _C_STD_CAN
from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig

# Windows / gs_usb 适配：piper_sdk 内部把 bustype 写死为 "socketcan"，
# 在 Windows 上需要改用 "gs_usb"（通过 Zadig + WinUSB + libusb）。
# 通过环境变量 LEROBOT_CAN_BUSTYPE / LEROBOT_CAN_CHANNEL / LEROBOT_CAN_BITRATE 控制：
#   - Linux (默认): LEROBOT_CAN_BUSTYPE=socketcan, can_name="can0"
#   - Windows:       LEROBOT_CAN_BUSTYPE=gs_usb,     can_name="0" (设备索引)
# monkey-patch C_STD_CAN，使其在 bustype=gs_usb 时：
#   1) 强制 bustype 为 gs_usb（覆盖 piper_sdk 写死的 socketcan）
#   2) 跳过 JudgeCanInfo（Linux 专属的 `ip link` 检查，Windows 上会报错）
#   3) 把 channel 从字符串转成 int（gs_usb 接口要求整数索引）
#   4) 用 interface= 而不是 bustype=（避免 python-can 4.2+ 弃用警告）
_LEROBOT_CAN_BUSTYPE = os.getenv("LEROBOT_CAN_BUSTYPE", "socketcan").lower()
_LEROBOT_CAN_BITRATE = int(os.getenv("LEROBOT_CAN_BITRATE", "1000000"))

if _LEROBOT_CAN_BUSTYPE == "gs_usb":
    _orig_C_STD_CAN_init = _C_STD_CAN.__init__
    _orig_C_STD_CAN_Init = _C_STD_CAN.Init

    def _patched_C_STD_CAN_init(self, channel_name="can0", bustype="socketcan",
                                expected_bitrate=1000000, judge_flag=True,
                                auto_init=True, callback_function=None):
        # gs_usb 接口要求 channel 为整数索引；允许传入 "0" 这样的字符串
        if isinstance(channel_name, str) and channel_name.isdigit():
            channel_name = int(channel_name)
        # 强制使用 gs_usb，覆盖 piper_sdk 写死的 socketcan
        # 跳过 JudgeCanInfo（Linux 专属的 `ip link` 检查）
        return _orig_C_STD_CAN_init(self, channel_name=channel_name,
                                    bustype=_LEROBOT_CAN_BUSTYPE,
                                    expected_bitrate=expected_bitrate,
                                    judge_flag=False,
                                    auto_init=auto_init,
                                    callback_function=callback_function)

    def _patched_C_STD_CAN_Init(self):
        if self.bus is not None:
            return
        import can as _can
        try:
            channel = self.channel_name
            if isinstance(channel, str) and channel.isdigit():
                channel = int(channel)
            self.bus = _can.interface.Bus(channel=channel,
                                          interface=_LEROBOT_CAN_BUSTYPE,
                                          bitrate=_LEROBOT_CAN_BITRATE)
            print(channel, "bus opened successfully (gs_usb).")
        except _can.CanError as e:
            print(f"Failed to open CAN bus: {e}")
            self.bus = None

    _C_STD_CAN.__init__ = _patched_C_STD_CAN_init
    _C_STD_CAN.Init = _patched_C_STD_CAN_Init
    print(f"[PiperMotorsBus] gs_usb 适配已启用 (channel 索引模式, bitrate={_LEROBOT_CAN_BITRATE})")


class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）
        self.joint_command_deadband = 0
        self.gripper_command_deadband = 0
        self._last_joint_command = None
        self._last_gripper_command = None
        self._last_move_mode = None
        self.interp_steps = max(1, int(os.getenv("LEROBOT_PIPER_INTERP_STEPS", "5")))
        self.interp_sleep_s = max(0.0, float(os.getenv("LEROBOT_PIPER_INTERP_SLEEP_S", "0.006")))
        # 执行速度 (1-100, 100=最大; 归位等大行程移动前调低可减少电机负载/发热)
        self.move_speed = max(1, min(100, int(os.getenv("LEROBOT_PIPER_MOVE_SPEED", "100"))))
        self._last_target_joint = list(self.init_joint_position)

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp
    
    def motor_names(self):
        return

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def hold_position(self, target_joint: list[float], hold_s: float = 2.0, interval_s: float = 0.05):
        """Repeatedly send a target for a short period so the arm actually settles there."""
        deadline = time.perf_counter() + max(0.0, hold_s)
        while time.perf_counter() < deadline:
            self.write(target_joint=target_joint)
            time.sleep(interval_s)

    def _send_joint_command(self, target_joint: list[float]) -> None:
        joint_0 = round(target_joint[0] * self.joint_factor)
        joint_1 = round(target_joint[1] * self.joint_factor)
        joint_2 = round(target_joint[2] * self.joint_factor)
        joint_3 = round(target_joint[3] * self.joint_factor)
        joint_4 = round(target_joint[4] * self.joint_factor)
        joint_5 = round(target_joint[5] * self.joint_factor)
        gripper_range = round(target_joint[6] * 1000 * 1000)

        joint_commands = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]

        if self._last_move_mode != "joint":
            self.piper.MotionCtrl_2(0x01, 0x01, self.move_speed, 0xAD) # joint control with MIT high-follow mode
            self._last_move_mode = "joint"

        self.piper.JointCtrl(*joint_commands)
        self.piper.GripperCtrl(gripper_range, 1000, 0x01, 0) # 单位 0.001°
        self._last_joint_command = joint_commands
        self._last_gripper_command = gripper_range

    def set_move_speed(self, speed: int) -> None:
        """设置执行速度 (1-100, 100=最大)。归位等大行程移动前调低可减少电机负载/发热。

        会强制下次 write 重发 MotionCtrl_2 以应用新速度。
        """
        self.move_speed = max(1, min(100, int(speed)))
        self._last_move_mode = None

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 (-92000~92000) / 57324.840764
                joint_2 (float): 关节2角度 -1300 ~ 90000 / 57324.840764
                joint_3 (float): 关节3角度 2400 ~ -80000 / 57324.840764
                joint_4 (float): 关节4角度 -90000~90000 / 57324.840764
                joint_5 (float): 关节5角度 19000~-77000 / 57324.840764
                joint_6 (float): 关节6角度 -90000~90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.1
        """
        target_joint = [float(value) for value in target_joint]
        start_joint = list(self._last_target_joint)
        steps = self.interp_steps

        if len(start_joint) != len(target_joint):
            start_joint = list(target_joint)

        for step_idx in range(1, steps + 1):
            ratio = step_idx / steps
            interp_target = [
                start + (target - start) * ratio
                for start, target in zip(start_joint, target_joint, strict=True)
            ]
            self._send_joint_command(interp_target)
            if step_idx < steps and self.interp_sleep_s > 0.0:
                time.sleep(self.interp_sleep_s)

        self._last_target_joint = list(target_joint)

    def write_pose(self, target_pose: list | tuple, gripper: float):
        """
            End-effector pose control
            - target_pose: [x, y, z, rx, ry, rz]
              x/y/z in meters, rx/ry/rz in degrees
            - gripper: opening in meters
        """
        if len(target_pose) != 6:
            raise ValueError(f"Expected 6D end-effector pose, got {len(target_pose)} values.")

        xyz_cmd = [round(axis * 1_000_000) for axis in target_pose[:3]]
        rpy_cmd = [round(axis * 1000) for axis in target_pose[3:]]
        gripper_range = round(gripper * 1000 * 1000)

        if self._last_move_mode != "pose":
            # SDK requires switching to end-pose control mode before EndPoseCtrl.
            self.piper.ModeCtrl(0x01, 0x00, 100, 0x00)
            self._last_move_mode = "pose"

        self.piper.EndPoseCtrl(*(xyz_cmd + rpy_cmd))

        gripper_changed = (
            self._last_gripper_command is None
            or abs(gripper_range - self._last_gripper_command) >= self.gripper_command_deadband
        )
        if gripper_changed:
            self.piper.GripperCtrl(gripper_range, 1000, 0x01, 0)
            self._last_gripper_command = gripper_range
    

    def read(self) -> Dict:
        """
            返回统一单位的机器人状态:
            - joints: radians
            - gripper: meters
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        
        return {
            "joint_1": joint_state.joint_1 / self.joint_factor,
            "joint_2": joint_state.joint_2 / self.joint_factor,
            "joint_3": joint_state.joint_3 / self.joint_factor,
            "joint_4": joint_state.joint_4 / self.joint_factor,
            "joint_5": joint_state.joint_5 / self.joint_factor,
            "joint_6": joint_state.joint_6 / self.joint_factor,
            "gripper": gripper_state.grippers_angle / 1_000_000,
        }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)
