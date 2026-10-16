"""
ATI Mini40 力/力矩传感器，通过 NI USB-6341 采集 + 标定矩阵计算。

连接方式: ATI Mini40 → 信号调理盒 → NI USB-6341 (ai0~ai5) → USB → PC

标定文件格式: ATI .cal (XML)，使用 UserAxis 6×6 矩阵将 6 路电压转为 Fx/Fy/Fz/Tx/Ty/Tz。

用法:
    sensor = ATIForceSensor(
        cal_path="path/to/FT32663.cal",
        device="Dev1",
        channels="ai0:5",
    )
    ft = sensor.read()  # np.array([Fx, Fy, Fz, Tx, Ty, Tz]) 单位: N / N-m
    sensor.close()
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

try:
    import nidaqmx
    _HAS_NIDAQMX = True
except ImportError:
    _HAS_NIDAQMX = False


class ATIForceSensor:
    """ATI Mini40 力传感器读取器（NI USB-6341 + nidaqmx）

    Parameters
    ----------
    cal_path : str
        ATI .cal 标定文件路径（XML 格式）
    device : str
        NI 设备名，如 "Dev1"
    channels : str
        模拟输入通道范围，如 "ai0:5"（6 路 gauge 电压）
    voltage_range : tuple[float, float]
        电压采集范围，默认 ±10V
    """

    def __init__(
        self,
        cal_path: str,
        device: str = "Dev1",
        channels: str = "ai0:5",
        voltage_range: tuple[float, float] = (-10.0, 10.0),
        mock: bool = False,
    ):
        self._mock = mock
        self._device = device
        self._channels = channels
        self._task = None
        self._is_connected = False

        # 解析标定矩阵
        if cal_path is None:
            cal_path = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
        if not cal_path or not os.path.isfile(cal_path):
            raise FileNotFoundError(f"标定文件不存在: {cal_path}")

        tree = ET.parse(cal_path)
        root = tree.getroot()
        self.serial = root.get("Serial")
        self.body_style = root.get("BodyStyle", "")

        cal = root.find("Calibration")
        if cal is None:
            raise ValueError("标定文件中未找到 <Calibration> 节点")
        self.part_number = cal.get("PartNumber", "")
        self.cal_date = cal.get("CalDate", "")
        self.force_units = cal.get("ForceUnits", "N")
        self.torque_units = cal.get("TorqueUnits", "N-m")

        # UserAxis 是给用户软件用的标定矩阵（Axis 是 ATI 内部用的）
        user_axes = root.findall(".//UserAxis")
        if len(user_axes) != 6:
            raise ValueError(f"UserAxis 数量应为 6，实际 {len(user_axes)}")
        self._cal_matrix = np.array([
            [float(v) for v in axis.get("values", "").split()]
            for axis in user_axes
        ])
        if self._cal_matrix.shape != (6, 6):
            raise ValueError(f"标定矩阵 shape 应为 (6,6)，实际 {self._cal_matrix.shape}")

        self._voltage_range = voltage_range

        # === 双段归零：基础偏置（传感器固有）+ 工具偏置（夹爪重量）===
        # 换工具时只需更新工具偏置，基础偏置保持不变
        base_path = os.path.splitext(cal_path)[0] + "_tare.npy"
        tool_path = os.path.splitext(cal_path)[0] + "_tool.npy"

        if os.path.exists(base_path):
            self._base_tare = np.load(base_path)
            print(f"[ATIForceSensor] 基础偏置: {np.round(self._base_tare, 2)}")
        else:
            self._base_tare = np.zeros(6)
            print(f"[ATIForceSensor] 无基础偏置（{os.path.basename(base_path)}）")

        if os.path.exists(tool_path):
            self._tool_tare = np.load(tool_path)
            print(f"[ATIForceSensor] 工具偏置: {np.round(self._tool_tare, 2)}")
        else:
            self._tool_tare = np.zeros(6)
            print(f"[ATIForceSensor] 无工具偏置（{os.path.basename(tool_path)}）")

        print(
            f"[ATIForceSensor] {self.serial} ({self.part_number}) "
            f"标定日期 {self.cal_date}"
        )

    def connect(self) -> None:
        """打开 NI-DAQmx 连续采样（1kHz，缓冲 60000 样本 = 60 秒）"""
        if self._is_connected:
            return
        if self._mock:
            self._is_connected = True
            return
        if not _HAS_NIDAQMX:
            raise ImportError("需要安装 nidaqmx: pip install nidaqmx")

        from nidaqmx.constants import AcquisitionType

        self._task = nidaqmx.Task()
        self._task.ai_channels.add_ai_voltage_chan(
            f"{self._device}/{self._channels}",
            min_val=self._voltage_range[0],
            max_val=self._voltage_range[1],
        )
        self._task.timing.cfg_samp_clk_timing(
            rate=1000,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=60000,
        )
        self._task.start()
        self._is_connected = True
        print(f"[ATIForceSensor] 已连接 {self._device}/{self._channels} (1kHz, 60s缓冲)")

    def read(self) -> np.ndarray:
        """排空缓冲取最新值，~0.5ms。"""
        if not self._is_connected:
            raise RuntimeError("力传感器未连接")
        if self._mock:
            return np.zeros(6)

        raw = self._task.read(number_of_samples_per_channel=-1)
        if not raw or len(raw[0]) == 0:
            return np.zeros(6)
        voltages = np.array([ch[-1] for ch in raw])
        return self._cal_matrix @ voltages - self._base_tare - self._tool_tare

    def drain(self) -> None:
        """排空缓冲（不返回数据，防止溢出用）。"""
        if not self._is_connected or self._mock:
            return
        try:
            self._task.read(number_of_samples_per_channel=-1)
        except Exception:
            pass

    def close(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
            except Exception:
                pass
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None
        self._is_connected = False

    def set_base_tare(self, n_samples: int = 100) -> np.ndarray:
        """无工具时归零（传感器固有偏置），保存到 _tare.npy"""
        vals = [self._read_raw() for _ in range(n_samples)]
        self._base_tare = np.mean(vals, axis=0)
        cal_path = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
        if cal_path:
            np.save(os.path.splitext(cal_path)[0] + "_tare.npy", self._base_tare)
        print(f"[ATIForceSensor] 基础偏置已保存: {np.round(self._base_tare, 2)}")
        return self._base_tare

    def set_tool_tare(self, n_samples: int = 100) -> np.ndarray:
        """装工具后归零（工具重量偏置），保存到 _tool.npy。
        读取时会减去 基础偏置 + 工具偏置，结果 = 接触力。
        """
        vals = [self._read_raw() for _ in range(n_samples)]
        # 工具偏置 = 当前读数 - 基础偏置
        raw_mean = np.mean(vals, axis=0)
        self._tool_tare = raw_mean - self._base_tare
        cal_path = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
        if cal_path:
            np.save(os.path.splitext(cal_path)[0] + "_tool.npy", self._tool_tare)
        print(f"[ATIForceSensor] 工具偏置已保存: {np.round(self._tool_tare, 2)}")
        return self._tool_tare

    def clear_tool_tare(self) -> None:
        """清除工具偏置"""
        self._tool_tare = np.zeros(6)
        cal_path = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
        if cal_path:
            tp = os.path.splitext(cal_path)[0] + "_tool.npy"
            if os.path.exists(tp):
                os.remove(tp)
        print("[ATIForceSensor] 工具偏置已清除")

    def _read_raw(self) -> np.ndarray:
        """读取原始 F/T（不减去偏置）"""
        raw = self._task.read(number_of_samples_per_channel=1)
        v = np.array([ch[0] for ch in raw])
        return self._cal_matrix @ v

    @property
    def base_tare(self) -> np.ndarray:
        return self._base_tare

    @property
    def tool_tare(self) -> np.ndarray:
        return self._tool_tare

    def is_connected(self) -> bool:
        return self._is_connected

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
