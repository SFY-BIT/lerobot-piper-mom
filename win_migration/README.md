# Windows 迁移：lerobot-piper-mom 环境配置

> 从 Linux (Ubuntu 22.04) 迁移到 Windows 的完整指南

---

## 迁移动机

NI USB-6341 在 Linux 上缺少固件支持，ATI Mini40 力传感器无法采集。
Windows 上 NI-DAQmx 原生支持 USB-6341，力传感器可即插即用。

---

## 一、硬件清单

| 硬件 | Linux 端口 | Windows 端口 | 备注 |
|------|-----------|-------------|------|
| Piper 机械臂 (CAN) | `can0` (gs_usb) | 需 Zadig 配置 | 见下方 CAN 配置 |
| UArm 主臂 (CH340) | `/dev/ttyUSB0` | `COM3` | 自动识别 |
| RealSense 435i | USB | USB | 自动识别 |
| Web 摄像头 | USB | USB | 自动识别 |
| NI USB-6341 | USB | USB | **NI-DAQmx 原生支持** |
| PS5 手柄 (可选) | 蓝牙 | 蓝牙 | 自动识别 |

---

## 二、CAN 适配器配置（最关键）

CAN 适配器型号：OpenMoko Geschwister Schneider (VID:1D50 PID:606F)

### 步骤

1. 下载 [Zadig](https://zadig.akeo.ie/)（USB 驱动安装工具）
2. 插上 CAN 适配器
3. 打开 Zadig → Options → List All Devices
4. 选中 CAN 适配器 → 选择 WinUSB (或 libusb-win32)
5. 点击 Install Driver

### 验证
```cmd
pip install python-can
python -c "import can; bus=can.Bus(bustype='gs_usb', channel='GS_USB_0', bitrate=1000000); print('OK')"
```

---

## 三、安装 Miniconda

下载 [Miniconda Windows x64](https://docs.conda.io/en/latest/miniconda.html)，安装后：

```cmd
conda init cmd
conda create -n lerobot python=3.10 -y
conda activate lerobot
```

---

## 四、安装 Python 依赖

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install pygame pyserial piper_sdk python-can scipy opencv-python
pip install pyrealsense2
pip install nidaqmx
pip install rerun-sdk
pip install jsonlines pyarrow tqdm "datasets<4.0.0"

pip install -e /path/to/lerobot-piper-mom
```

---

## 五、安装 NI-DAQmx（ATI 力传感器）

1. 下载 [NI-DAQmx 2026 Q2 for Windows](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html)
2. 安装（默认选项即可）
3. 验证：
```cmd
nilsdev
:: 应输出: NI USB-6341
```

---

## 六、代码修改

### 1. CAN 端口 (`lerobot-piper-mom` 相关配置)

**所有用到 `can0` 的地方**改为 `GS_USB_0`（或 `bustype='gs_usb'`）：

```python
# 修改前
can_name="can0"
bustype="socketcan"

# 修改后
can_name="GS_USB_0"
bustype="gs_usb"
```

或设置环境变量：
```cmd
set LEROBOT_CAN_CHANNEL=GS_USB_0
set LEROBOT_CAN_BUSTYPE=gs_usb
```

### 2. UArm 串口 (`control_robot.py` 等)

```python
# 修改前
LEROBOT_UARM_PORT=/dev/ttyUSB0

# 修改后（Windows）
LEROBOT_UARM_PORT=COM3
```

### 3. Evo-RL 遥操作命令

```cmd
python -m lerobot.scripts.lerobot_teleoperate ^
  --robot.type=piper_follower --robot.port=GS_USB_0 ^
  --robot.cameras='{...}' ^
  --teleop.type=uarm_leader --teleop.port=COM3 ^
  --fps=30 --display_data=true
```

---

## 七、ATI 力传感器采集

Windows 上直接用 NI-DAQmx Python API：

```python
import nidaqmx
import numpy as np

task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5", min_val=-10.0, max_val=10.0)
data = task.read(number_of_samples_per_channel=1)
# data = [Fx_voltage, Fy_voltage, Fz_voltage, Tx_voltage, Ty_voltage, Tz_voltage]

# 应用标定矩阵（从 .cal 文件解析）
ft = cal_matrix @ np.array(data)  # 6x6矩阵 × 6路电压
```

---

## 八、验证清单

- [ ] CAN 适配器在 Zadig 中显示正确驱动
- [ ] `python -c "import can; can.Bus(bustype='gs_usb', ...)"` 不报错
- [ ] `conda activate lerobot && python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `nilsdev` 输出 `NI USB-6341`
- [ ] `python -c "import nidaqmx"` 不报错
- [ ] UArm 串口可在设备管理器看到 COM3
- [ ] 摄像头在设备管理器正常
- [ ] 遥操作启动命令正常运行
